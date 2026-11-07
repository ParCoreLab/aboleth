#include "aboleth/Transforms/Passes.h"

#include "aboleth/Target/RdmaAbi.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHFORWARDINLINEPAYLOADSPASS
#include "aboleth/Transforms/Passes.h.inc"

namespace {

struct ForwardInlinePayloads
    : public impl::AbolethForwardInlinePayloadsPassBase<
          ForwardInlinePayloads> {
  using Base::Base;

  void runOnOperation() override;
};

// A pointer resolved to a base value plus a constant byte displacement. Two
// addresses are comparable only when they share a base: this pass proves
// aliasing facts from that shared base and never from raw address equality.
struct Address {
  Value base;
  int64_t offset;
};

// Bytes one scalar occupies, or nothing when it is not byte-addressable data
// this pass can move into a payload. The accepted widths are exactly those the
// inline payload defines a packing for (see verifyInlinePayload): a value of
// any other width stays in its buffer, which only costs the transfer its
// forwarding.
std::optional<int64_t> scalarBytes(Type type) {
  if (!type.isIntOrFloat())
    return std::nullopt;
  unsigned width = type.getIntOrFloatBitWidth();
  if (width != 8 && width != 16 && width != 24 && width != 32 && width != 64)
    return std::nullopt;
  return width / 8;
}

// The same for a Triton pointer's pointee, which is the stride an addptr step
// scales by. A tensor of pointers has no single stride and resolves to nothing.
std::optional<int64_t> pointeeBytes(Type type) {
  auto pointer = dyn_cast<triton::PointerType>(type);
  if (!pointer)
    return std::nullopt;
  return scalarBytes(pointer.getPointeeType());
}

// A constant integer operand. MLIR's `m_ConstantInt` asks whether the defining
// operation carries the ConstantLike trait, and a trait's TypeID is not shared
// across the plugin boundary, so every constant would look non-constant here.
// The plugin matches the constant operation itself instead.
std::optional<int64_t> constantOffset(Value value) {
  if (!isa<IntegerType>(value.getType()))
    return std::nullopt;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

// A pointer bitcast changes how bytes are typed, not where they live, so it is
// transparent to address resolution. Mixed-width staging goes through one.
Value bitcastSource(Value pointer) {
  auto cast = pointer.getDefiningOp<triton::BitcastOp>();
  if (!cast || !isa<triton::PointerType>(cast.getSrc().getType()) ||
      !isa<triton::PointerType>(cast.getResult().getType()))
    return {};
  return cast.getSrc();
}

// Walk a `tt.addptr` chain with constant offsets down to the value it advances
// from. A non-constant step, a tensor of pointers, or an opaque pointee leaves
// the address unresolved and the put unforwarded.
std::optional<Address> resolveAddress(Value pointer) {
  int64_t offset = 0;
  llvm::DenseSet<Value> seen;
  while (seen.insert(pointer).second) {
    if (auto advance = pointer.getDefiningOp<triton::AddPtrOp>()) {
      auto elementBytes = pointeeBytes(advance.getPtr().getType());
      auto step = constantOffset(advance.getOffset());
      if (!elementBytes || !step)
        return std::nullopt;
      // Triton scales an addptr step by the pointee size.
      offset += *step * *elementBytes;
      pointer = advance.getPtr();
      continue;
    }
    if (Value source = bitcastSource(pointer)) {
      pointer = source;
      continue;
    }
    return Address{pointer, offset};
  }
  return std::nullopt;
}

// One scalar store that may contribute payload bytes.
struct Contribution {
  triton::StoreOp store;
  Value base;
  int64_t offset;
  int64_t bytes;
};

// A store this pass can reason about: one scalar value, no mask, and an address
// that resolves to a constant offset from a base.
std::optional<Contribution> asScalarStore(Operation *operation) {
  auto store = dyn_cast<triton::StoreOp>(operation);
  if (!store || store.getMask())
    return std::nullopt;
  auto bytes = scalarBytes(store.getValue().getType());
  auto address = resolveAddress(store.getPtr());
  if (!bytes || !address)
    return std::nullopt;
  return Contribution{store, address->base, address->offset, *bytes};
}

// The byte range an operation reads or writes through `base`, or nothing when
// the operation neither touches `base` nor is provably unrelated to it.
struct Access {
  int64_t offset;
  int64_t bytes;
};

std::optional<Access> resolvedAccess(Operation *operation, Value base) {
  Value pointer;
  Type element;
  if (auto store = dyn_cast<triton::StoreOp>(operation)) {
    if (store.getMask())
      return std::nullopt;
    pointer = store.getPtr();
    element = store.getValue().getType();
  } else if (auto load = dyn_cast<triton::LoadOp>(operation)) {
    if (load.getMask())
      return std::nullopt;
    pointer = load.getPtr();
    element = load.getResult().getType();
  } else {
    return std::nullopt;
  }
  auto bytes = scalarBytes(element);
  auto address = resolveAddress(pointer);
  if (!bytes || !address || address->base != base)
    return std::nullopt;
  return Access{address->offset, *bytes};
}

bool overlaps(int64_t firstOffset, int64_t firstBytes, int64_t secondOffset,
              int64_t secondBytes) {
  return firstOffset < secondOffset + secondBytes &&
         secondOffset < firstOffset + firstBytes;
}

// Forwarding deletes the staging stores, so the forwarded byte range must be
// unreachable from anything else: a range some other operation can still read
// is live storage, not a staging slot. Sibling slots inside one buffer are
// allowed, because they resolve to disjoint ranges of the same base. Every
// operation whose access cannot be resolved is treated as reaching the range.
bool rangeIsExclusive(Value base, int64_t start, int64_t bytes,
                      const llvm::DenseSet<Operation *> &permitted) {
  SmallVector<Value> worklist{base};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value pointer = worklist.pop_back_val();
    if (!visited.insert(pointer).second)
      continue;
    for (Operation *user : pointer.getUsers()) {
      if (permitted.contains(user))
        continue;
      if (auto advance = dyn_cast<triton::AddPtrOp>(user)) {
        if (advance.getPtr() != pointer)
          return false;
        worklist.push_back(advance.getResult());
        continue;
      }
      if (auto cast = dyn_cast<triton::BitcastOp>(user)) {
        if (!bitcastSource(cast.getResult()))
          return false;
        worklist.push_back(cast.getResult());
        continue;
      }
      // Another slot's source binding: the range it transfers is the byte count
      // of every put that consumes it.
      if (auto binding = dyn_cast<BindLocalRegionOp>(user)) {
        auto address = resolveAddress(binding.getPointer());
        if (!address || address->base != base)
          return false;
        for (Operation *consumer : binding.getRegion().getUsers()) {
          auto other = dyn_cast<PutOp>(consumer);
          if (!other)
            return false;
          auto count = constantOffset(other.getBytes());
          if (!count || *count <= 0 ||
              overlaps(address->offset, *count, start, bytes))
            return false;
        }
        continue;
      }
      auto access = resolvedAccess(user, base);
      if (!access || overlaps(access->offset, access->bytes, start, bytes))
        return false;
    }
  }
  return true;
}

// The pure address arithmetic that feeds `pointer`, collected before anything is
// erased so no walk ever follows a value whose defining operation is gone.
void collectAddressChain(Value pointer,
                         llvm::SmallSetVector<Operation *, 8> &chain) {
  while (Operation *producer = pointer.getDefiningOp()) {
    if (!isa<triton::AddPtrOp, triton::BitcastOp>(producer) ||
        !chain.insert(producer))
      return;
    pointer = producer->getOperand(0);
  }
}

// Retire whatever of that arithmetic the removed operations were its only
// reason to exist. Each one is pure, so a use-free one carries no effect and
// only clutters the module.
void eraseDeadAddresses(const llvm::SmallSetVector<Operation *, 8> &chain) {
  SmallVector<Operation *> pending(chain.begin(), chain.end());
  bool progressed = true;
  while (progressed) {
    progressed = false;
    for (Operation *&operation : pending) {
      if (!operation || !operation->use_empty())
        continue;
      operation->erase();
      operation = nullptr;
      progressed = true;
    }
  }
}

// Rewrite one put into an inline transfer, or leave it alone. Every bail-out is
// a fact the pass could not prove, never a judgement about profitability.
bool forwardPut(PutOp put, int64_t inlineBytes) {
  auto binding = put.getSource().getDefiningOp<BindLocalRegionOp>();
  if (!binding)
    return false;
  auto byteCount = constantOffset(put.getBytes());
  if (!byteCount || *byteCount <= 0 || *byteCount > inlineBytes)
    return false;
  auto source = resolveAddress(binding.getPointer());
  if (!source)
    return false;

  Block *block = put->getBlock();
  const int64_t start = source->offset;
  const int64_t end = start + *byteCount;

  // Candidate stores are the scalar stores to this base that precede the put in
  // its own block. Dominance inside one block is program order, and the rest of
  // the batching contract is already single-block.
  SmallVector<Contribution> contributions;
  for (Operation &operation : *block) {
    if (&operation == put.getOperation())
      break;
    auto contribution = asScalarStore(&operation);
    if (!contribution || contribution->base != source->base)
      continue;
    if (!overlaps(contribution->offset, contribution->bytes, start,
                  end - start))
      continue;
    // A store that leaves the transferred range, or one that repeats a range an
    // earlier store already covered, is not a tiling this pass can order.
    if (contribution->offset < start ||
        contribution->offset + contribution->bytes > end)
      return false;
    if (llvm::any_of(contributions, [&](const Contribution &other) {
          return overlaps(other.offset, other.bytes, contribution->offset,
                          contribution->bytes);
        }))
      return false;
    contributions.push_back(*contribution);
  }
  if (contributions.empty())
    return false;

  // The stores must tile the transferred range exactly: no gap the buffer would
  // still have to supply, and no byte left to a previous generation.
  SmallVector<Contribution> ordered = contributions;
  llvm::sort(ordered, [](const Contribution &first,
                         const Contribution &second) {
    return first.offset < second.offset;
  });
  int64_t cursor = start;
  for (const Contribution &contribution : ordered) {
    if (contribution.offset != cursor)
      return false;
    cursor += contribution.bytes;
  }
  if (cursor != end)
    return false;

  // Nothing between the last contributing store and the put may touch the
  // range. An operation is crossed only when it has no memory effects or when
  // it provably accesses a disjoint part of the same base.
  Operation *lastStore = contributions.back().store;
  for (Operation *operation = lastStore->getNextNode();
       operation && operation != put.getOperation();
       operation = operation->getNextNode()) {
    if (isMemoryEffectFree(operation))
      continue;
    auto access = resolvedAccess(operation, source->base);
    if (access && !overlaps(access->offset, access->bytes, start, end - start))
      continue;
    return false;
  }

  llvm::DenseSet<Operation *> permitted{binding.getOperation()};
  for (const Contribution &contribution : ordered)
    permitted.insert(contribution.store);
  if (!rangeIsExclusive(source->base, start, end - start, permitted))
    return false;

  OpBuilder builder(put);
  SmallVector<Value> values;
  values.reserve(ordered.size());
  for (Contribution &contribution : ordered) {
    Value value = contribution.store.getValue();
    if (isa<IntegerType>(value.getType())) {
      values.push_back(value);
      continue;
    }
    // A float payload travels as its own bits; the transport copies bytes and
    // never interprets them.
    Type bits = builder.getIntegerType(
        value.getType().getIntOrFloatBitWidth());
    values.push_back(
        arith::BitcastOp::create(builder, value.getLoc(), bits, value));
  }

  llvm::SmallSetVector<Operation *, 8> addresses;
  collectAddressChain(binding.getPointer(), addresses);
  for (Contribution &contribution : ordered)
    collectAddressChain(contribution.store.getPtr(), addresses);

  auto forwarded = PutInlineOp::create(
      builder, put.getLoc(), put.getToken().getType(), put.getEndpoint(),
      put.getDestination(), values, put.getPredicate(), put.getAfter());
  put.getToken().replaceAllUsesWith(forwarded.getToken());
  put.erase();
  for (Contribution &contribution : ordered)
    contribution.store.erase();
  if (binding.getRegion().use_empty())
    binding.erase();
  eraseDeadAddresses(addresses);
  return true;
}

} // namespace

// TODO: the cap is one fixed option. Make it a position-dependent cost
// decision: one WQEBB inside throughput-bound batches, up to ~90 bytes for the
// command that ends a latency-critical chain.
void ForwardInlinePayloads::runOnOperation() {
  if (inlineBytes <= 0)
    return;
  // The cap decides which puts become inline commands, so a cap the wire
  // cannot carry would build corrupt work queue entries out of legal IR.
  if (static_cast<uint64_t>(inlineBytes) >
      ::aboleth::abi::kMaxInlinePayloadBytes) {
    getOperation()->emitError("inline payload cap of ")
        << inlineBytes << " bytes exceeds the "
        << ::aboleth::abi::kMaxInlinePayloadBytes
        << "-byte maximum one work queue entry can carry";
    signalPassFailure();
    return;
  }
  SmallVector<PutOp> puts;
  getOperation().walk([&](PutOp put) { puts.push_back(put); });
  // Program order, so a chained put sees the token its predecessor now
  // produces before it is examined itself.
  for (PutOp put : puts)
    forwardPut(put, inlineBytes);
}

} // namespace mlir::aboleth
