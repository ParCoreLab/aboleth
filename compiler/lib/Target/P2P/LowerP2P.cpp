#include "aboleth/Target/Passes.h"
#include "aboleth/Target/QueueLowering.h"
#include "aboleth/Target/TritonBridge.h"
#include "aboleth/Target/TritonEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHLOWERP2PPASS
#include "aboleth/Target/Passes.h.inc"

namespace {

struct LowerP2P : public impl::AbolethLowerP2PPassBase<LowerP2P> {
  using Base::Base;

  void runOnOperation() override;
};

using emit::addressOffset;
using emit::i32Constant;
using emit::packInlinePayload;
using emit::pointerAddress;
using emit::pointerFromAddress;
using emit::store32;

void emitCopyLoop(OpBuilder &builder, Location loc, Value sourceAddress,
                  Value destinationAddress, Value bytes, std::int32_t step,
                  Type elementType, bool perThread = false,
                  Value start = {}) {
  if (!start)
    start = arith::ConstantOp::create(
        builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  Value increment = arith::ConstantOp::create(
      builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(step));
  auto loop =
      scf::ForOp::create(builder, loc, start, bytes, increment);
  builder.setInsertionPoint(loop.getBody()->getTerminator());
  Value offset = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), loop.getInductionVar());
  Value source = pointerFromAddress(
      builder, loc,
      arith::AddIOp::create(builder, loc, sourceAddress, offset), elementType);
  Value destination = pointerFromAddress(
      builder, loc,
      arith::AddIOp::create(builder, loc, destinationAddress, offset),
      elementType);
  Value value = triton::LoadOp::create(
      builder, loc, source, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, /*isVolatile=*/false);
  if (perThread) {
    // Triton emits a scalar tt.store once per block, from lane 0; a
    // detached fragment's calling thread owns its own copy, so the store
    // must land from every caller.
    if (step == 8) {
      emit::storeThrough64(builder, loc, destination, value,
                           /*perThread=*/true);
    } else {
      Value widened = arith::ExtUIOp::create(
          builder, loc, builder.getI32Type(), value);
      triton::ElementwiseInlineAsmOp::create(
          builder, loc, TypeRange{builder.getI32Type()},
          "{ mov.b32 $0, $2; st.global.b8 [$1], $2; }",
          "=r,l,r", /*isPure=*/false, /*pack=*/1,
          ValueRange{destination, widened});
    }
    return;
  }
  triton::StoreOp::create(builder, loc, destination, value,
                          triton::CacheModifier::NONE,
                          triton::EvictionPolicy::NORMAL);
}

// A detached caller's 8-byte-aligned copy. The 8-byte loop serializes one
// load and one store per word, so a lone thread moving kilobytes over NVLink
// is load-latency bound; when both addresses are 16-byte aligned, the body
// instead moves 64 bytes per iteration as four 16-byte loads in flight
// before their four stores, and the 8-byte loop takes the tail.
void emitDetachedWideCopy(OpBuilder &builder, Location loc,
                          Value sourceAddress, Value destinationAddress,
                          Value bytes) {
  constexpr std::int32_t kChunkBytes = 64;
  Value combined = arith::OrIOp::create(builder, loc, sourceAddress,
                                        destinationAddress);
  Value misalignment = arith::AndIOp::create(
      builder, loc, combined,
      arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                builder.getI64IntegerAttr(15)));
  Value aligned16 = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, misalignment,
      arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                builder.getI64IntegerAttr(0)));
  auto conditional = scf::IfOp::create(builder, loc, TypeRange{}, aligned16,
                                       /*withElseRegion=*/true);
  builder.setInsertionPoint(conditional.thenBlock()->getTerminator());
  Value body = arith::AndIOp::create(
      builder, loc, bytes,
      arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                builder.getI32IntegerAttr(~(kChunkBytes - 1))));
  {
    OpBuilder::InsertionGuard guard(builder);
    auto loop = scf::ForOp::create(
        builder, loc,
        arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                  builder.getI32IntegerAttr(0)),
        body,
        arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                  builder.getI32IntegerAttr(kChunkBytes)));
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    Value offset = arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                          loop.getInductionVar());
    Value source = arith::AddIOp::create(builder, loc, sourceAddress, offset);
    Value destination =
        arith::AddIOp::create(builder, loc, destinationAddress, offset);
    // Every caller runs its own copy (Triton's scalar tt.store would land
    // from lane 0 only): per-thread inline asm, stores as storeThrough64's.
    triton::ElementwiseInlineAsmOp::create(
        builder, loc, TypeRange{builder.getI32Type()},
        "{ .reg .b64 w<8>; "
        "ld.global.v2.u64 {w0, w1}, [$1]; "
        "ld.global.v2.u64 {w2, w3}, [$1+16]; "
        "ld.global.v2.u64 {w4, w5}, [$1+32]; "
        "ld.global.v2.u64 {w6, w7}, [$1+48]; "
        "st.global.L1::no_allocate.v2.u64 [$2], {w0, w1}; "
        "st.global.L1::no_allocate.v2.u64 [$2+16], {w2, w3}; "
        "st.global.L1::no_allocate.v2.u64 [$2+32], {w4, w5}; "
        "st.global.L1::no_allocate.v2.u64 [$2+48], {w6, w7}; "
        "mov.b32 $0, 0; }",
        "=r,l,l", /*isPure=*/false, /*pack=*/1,
        ValueRange{source, destination});
  }
  emitCopyLoop(builder, loc, sourceAddress, destinationAddress, bytes,
               /*step=*/8, builder.getI64Type(), /*perThread=*/true, body);
  builder.setInsertionPoint(conditional.elseBlock()->getTerminator());
  emitCopyLoop(builder, loc, sourceAddress, destinationAddress, bytes,
               /*step=*/8, builder.getI64Type(), /*perThread=*/true);
}

bool forceScalarCopies() {
  const char *configured = std::getenv("ABOLETH_P2P_SCALAR");
  if (configured == nullptr || configured[0] == '\0')
    return false;
  llvm::StringRef value(configured);
  return value != "0" && !value.equals_insensitive("false") &&
         !value.equals_insensitive("off");
}

bool cooperativeCopiesEnabled() {
  const char *configured = std::getenv("ABOLETH_P2P_COOP");
  if (configured == nullptr || configured[0] == '\0')
    return true;
  llvm::StringRef value(configured);
  return value != "0" && !value.equals_insensitive("false") &&
         !value.equals_insensitive("off");
}

std::int32_t cooperativeMinimumBytes() {
  constexpr std::int32_t fallback = 256;
  const char *configured = std::getenv("ABOLETH_P2P_COOP_MIN");
  if (configured == nullptr || configured[0] == '\0')
    return fallback;
  char *end = nullptr;
  long value = std::strtol(configured, &end, 10);
  if (end == configured || *end != '\0' || value < 8 ||
      value > std::numeric_limits<std::int32_t>::max() - 7)
    return fallback;
  return static_cast<std::int32_t>((value + 7) & ~7L);
}

// The cooperative chunk, in 8-byte elements. Each iteration's loads feed
// its stores, so the chunk is the copy's memory-level parallelism: 2048
// elements (16 KiB; 64 per lane of a one-warp launch) keep enough loads in
// flight to run a multi-MiB NVLink copy at link rate, where 256 left it
// load-latency bound (3x slower at 4 MiB), and 4096
// spills. When the payload size is a compile-time constant (Tier 2 bakes
// it) the chunk shrinks to the smallest power of two covering it, so a
// small put carries no masked-off lanes. ABOLETH_P2P_COOP_ELEMENTS (a power
// of two in [32, 4096], the planner's choice included) overrides both.
std::int32_t cooperativeElements(Value bytes) {
  constexpr std::int32_t kDefault = 2048;
  constexpr std::int32_t kMinimum = 32;
  if (const char *configured = std::getenv("ABOLETH_P2P_COOP_ELEMENTS");
      configured != nullptr && configured[0] != '\0') {
    char *end = nullptr;
    long value = std::strtol(configured, &end, 10);
    if (end != configured && *end == '\0' && value >= kMinimum &&
        value <= 4096 && (value & (value - 1)) == 0)
      return static_cast<std::int32_t>(value);
  }
  // The constant operation itself, not m_ConstantInt: a trait's TypeID is
  // not shared across the plugin boundary (ForwardInlinePayloads.cpp).
  auto constant = bytes.getDefiningOp<arith::ConstantOp>();
  auto integer =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  if (!integer)
    return kDefault;
  const std::uint64_t words = (integer.getValue().getZExtValue() + 7) / 8;
  std::int32_t elements = kMinimum;
  while (elements < kDefault && static_cast<std::uint64_t>(elements) < words)
    elements *= 2;
  return elements;
}

void emitCooperativeCopy(OpBuilder &builder, Location loc,
                         Value sourceAddress, Value destinationAddress,
                         Value bytes) {
  // TTIR-to-TTGIR distributes each chunk across the launch warps, while the
  // mask handles a final partial chunk. The chunk is tunable because large MoE
  // payloads need substantially more memory-level parallelism than signals.
  const std::int32_t elements = cooperativeElements(bytes);
  constexpr std::int32_t kElementBytes = 8;
  const std::int32_t blockBytes =
      elements * kElementBytes;

  Value zero = arith::ConstantOp::create(
      builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  Value increment = arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getI32IntegerAttr(blockBytes));
  auto loop =
      scf::ForOp::create(builder, loc, zero, bytes, increment);
  builder.setInsertionPoint(loop.getBody()->getTerminator());

  auto indexType =
      RankedTensorType::get({elements}, builder.getI32Type());
  Value indices = triton::MakeRangeOp::create(
      builder, loc, indexType, 0, elements);
  Value elementBytes = arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getI32IntegerAttr(kElementBytes));
  Value elementBytesTensor =
      triton::SplatOp::create(builder, loc, indexType, elementBytes);
  Value byteOffsets = arith::MulIOp::create(
      builder, loc, indexType, indices, elementBytesTensor);
  Value remaining = arith::SubIOp::create(
      builder, loc, bytes, loop.getInductionVar());
  Value remainingTensor =
      triton::SplatOp::create(builder, loc, indexType, remaining);
  auto maskType =
      RankedTensorType::get({elements}, builder.getI1Type());
  Value mask = arith::CmpIOp::create(
      builder, loc, maskType, arith::CmpIPredicate::ult,
      byteOffsets, remainingTensor);

  Value chunkElements = arith::DivUIOp::create(
      builder, loc, loop.getInductionVar(), elementBytes);
  Value chunkElementsTensor =
      triton::SplatOp::create(builder, loc, indexType, chunkElements);
  Value elementOffsets = arith::AddIOp::create(
      builder, loc, indexType, chunkElementsTensor, indices);

  Type elementType = builder.getI64Type();
  auto pointerType = triton::getPointerType(elementType);
  auto pointerTensorType =
      RankedTensorType::get({elements}, pointerType);
  Value source = pointerFromAddress(
      builder, loc, sourceAddress, elementType);
  Value destination = pointerFromAddress(
      builder, loc, destinationAddress, elementType);
  Value sourcePointers =
      triton::SplatOp::create(builder, loc, pointerTensorType, source);
  Value destinationPointers =
      triton::SplatOp::create(builder, loc, pointerTensorType,
                              destination);
  sourcePointers = triton::AddPtrOp::create(
      builder, loc, pointerTensorType, sourcePointers, elementOffsets);
  destinationPointers = triton::AddPtrOp::create(
      builder, loc, pointerTensorType, destinationPointers,
      elementOffsets);
  Value values = triton::LoadOp::create(
      builder, loc, sourcePointers, mask, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, /*isVolatile=*/false);
  triton::StoreOp::create(
      builder, loc, destinationPointers, values, mask,
      triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
      /*ignoreCta=*/false);
}

// The one-warp form: a detached fragment may not impose a block barrier on
// its caller, so the cooperative copy's fence synchronizes the calling warp
// only — the caller's contract is to enter with all 32 lanes of one warp.
void emitWarpFence(Operation *insertionPoint) {
  OpBuilder builder(insertionPoint);
  OperationState state(
      insertionPoint->getLoc(),
      triton::ElementwiseInlineAsmOp::getOperationName());
  state.addAttribute(
      "asm_string",
      builder.getStringAttr(
          "bar.warp.sync 0xffffffff;\nmembar.sys;\nmov.u32 $0, 0;"));
  state.addAttribute(
      "constraints", builder.getStringAttr("=r"));
  state.addAttribute("pure", builder.getBoolAttr(false));
  state.addAttribute(
      "packed_element", builder.getI32IntegerAttr(1));
  state.addTypes(builder.getI32Type());
  builder.create(state);
}

// The cooperative copy's ordering. bar.sync orders every thread's tile
// stores before what the CTA does next; when a signal follows in the same
// submission, its release-system exchange is cumulative over those stores
// (they happen before it through the barrier), so it publishes the tile to a
// peer-GPU acquirer and the copy needs no system fence of its own. Without a
// later signal the copy keeps membar.sys. A loopback peer on the same device
// needs only GPU scope.
void emitCooperativeFence(Operation *insertionPoint, bool systemFence) {
  OpBuilder builder(insertionPoint);
  if (emit::hipDeviceLibrary(builder)) {
    triton::ElementwiseInlineAsmOp::create(
        builder, insertionPoint->getLoc(), TypeRange{builder.getI32Type()},
        "s_barrier\n\tv_mov_b32 $0, 0", "=v", /*isPure=*/false, /*pack=*/1, ValueRange{});
    if (systemFence)
      emit::hipCall(builder, insertionPoint->getLoc(), builder.getI64Type(),
                    "aboleth_hip_fence", {});
    return;
  }
  OperationState state(
      insertionPoint->getLoc(),
      triton::ElementwiseInlineAsmOp::getOperationName());
  state.addAttribute(
      "asm_string",
      builder.getStringAttr(
          systemFence ? "bar.sync 0;\nmembar.sys;\nmov.u32 $0, 0;"
                      : "bar.sync 0;\nmov.u32 $0, 0;"));
  state.addAttribute(
      "constraints", builder.getStringAttr("=r"));
  state.addAttribute("pure", builder.getBoolAttr(false));
  state.addAttribute(
      "packed_element", builder.getI32IntegerAttr(1));
  state.addTypes(builder.getI32Type());
  // Triton's inline-asm conversion requires a result. The dummy move
  // keeps the side-effecting barrier/fence representable through TTIR.
  builder.create(state);
}

LogicalResult lowerPostPut(Operation *insertionPoint, PostPutOp op,
                           bool detached, llvm::StringRef copyMode,
                           bool publishedBySignal) {
  OpBuilder builder(insertionPoint);
  Location loc = op.getLoc();
  auto source = materializeLocalRegion(op.getSource(), op);
  auto destination = materializeRemoteRegion(op.getDestination(), op);
  if (failed(source) || failed(destination))
    return failure();
  auto conditional = scf::IfOp::create(
      builder, loc, TypeRange{}, op.getPredicate(),
      /*withElseRegion=*/false);
  builder.setInsertionPoint(
      conditional.thenBlock()->getTerminator());
  Value sourceAddress =
      pointerAddress(builder, loc, source->pointer);
  Value destinationAddress =
      pointerAddress(builder, loc, destination->pointer);

  // An explicit copy mode is the generator's decision (a detached fragment
  // may not read build-machine environment); empty keeps the in-kernel
  // env-driven behaviour, with 'wide' as the detached fallback because the
  // cooperative path's block fence is illegal in a detached function.
  std::string mode = copyMode.str();
  if (mode.empty() && detached)
    mode = "wide";

  if (detached && emit::hipDeviceLibrary(builder)) {
    emit::hipCall(builder, loc, builder.getI64Type(), "aboleth_hip_copy",
                  {destinationAddress, sourceAddress, op.getBytes(),
                   i32Constant(builder, loc, mode == "warp")});
    return success();
  }

  if (mode == "scalar" || (mode.empty() && forceScalarCopies())) {
    emitCopyLoop(builder, loc, sourceAddress, destinationAddress, op.getBytes(),
                 /*step=*/1, builder.getI8Type(), detached);
    return success();
  }

  Value bytes64 = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), op.getBytes());
  Value combined = arith::OrIOp::create(
      builder, loc,
      arith::OrIOp::create(
          builder, loc, sourceAddress, destinationAddress),
      bytes64);
  Value alignmentMask = arith::ConstantOp::create(
      builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(7));
  Value misalignment =
      arith::AndIOp::create(builder, loc, combined, alignmentMask);
  Value zero64 = arith::ConstantOp::create(
      builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
  Value aligned = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, misalignment, zero64);

  auto widthConditional = scf::IfOp::create(
      builder, loc, TypeRange{}, aligned, /*withElseRegion=*/true);
  builder.setInsertionPoint(
      widthConditional.thenBlock()->getTerminator());
  if (mode == "warp") {
    // One-warp cooperative copy: the tensor distributes over the calling
    // warp's 32 lanes and the fence synchronizes that warp only. The
    // caller's contract is to enter with a full warp.
    emitCooperativeCopy(builder, loc, sourceAddress,
                        destinationAddress, op.getBytes());
    emitWarpFence(
        widthConditional.thenBlock()->getTerminator());
  } else if (mode == "wide" && detached) {
    emitDetachedWideCopy(builder, loc, sourceAddress, destinationAddress,
                         op.getBytes());
  } else if (mode == "wide") {
    emitCopyLoop(builder, loc, sourceAddress, destinationAddress,
                 op.getBytes(), /*step=*/8, builder.getI64Type(), detached);
  } else if (cooperativeCopiesEnabled()) {
    Value minimum = arith::ConstantOp::create(
        builder, loc, builder.getI32Type(),
        builder.getI32IntegerAttr(cooperativeMinimumBytes()));
    Value cooperative = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, op.getBytes(), minimum);
    auto sizeConditional = scf::IfOp::create(
        builder, loc, TypeRange{}, cooperative,
        /*withElseRegion=*/true);
    builder.setInsertionPoint(
        sizeConditional.thenBlock()->getTerminator());
    emitCooperativeCopy(builder, loc, sourceAddress,
                        destinationAddress, op.getBytes());
    emitCooperativeFence(
        sizeConditional.thenBlock()->getTerminator(),
        /*systemFence=*/!publishedBySignal);
    builder.setInsertionPoint(
        sizeConditional.elseBlock()->getTerminator());
    emitCopyLoop(builder, loc, sourceAddress, destinationAddress,
                 op.getBytes(), /*step=*/8, builder.getI64Type(), detached);
  } else {
    emitCopyLoop(builder, loc, sourceAddress, destinationAddress,
                 op.getBytes(), /*step=*/8, builder.getI64Type(), detached);
  }
  builder.setInsertionPoint(
      widthConditional.elseBlock()->getTerminator());
  emitCopyLoop(builder, loc, sourceAddress, destinationAddress, op.getBytes(),
               /*step=*/1, builder.getI8Type(), detached);
  return success();
}

LogicalResult lowerPostSignal(Operation *insertionPoint, PostSignalOp op,
                              bool detached) {
  OpBuilder builder(insertionPoint);
  Location loc = op.getLoc();
  auto signalValue = materializeSignalValue(builder, op.getSource(), op);
  auto destination = materializeRemoteRegion(op.getDestination(), op);
  if (failed(signalValue) || failed(destination))
    return failure();
  auto destinationType =
      dyn_cast<triton::PointerType>(destination->pointer.getType());
  if (!destinationType || !destinationType.getPointeeType().isInteger(64))
    return op.emitError("P2P signals require an i64 destination");

  Value value = *signalValue;
  if (emit::hipDeviceLibrary(builder) && op.getExpected())
    return op.emitError("HIP P2P cannot realize an explicitly checked CAS publication");
  if (detached && emit::hipDeviceLibrary(builder)) {
    emit::releaseStore64(builder, loc, destination->pointer, value,
                          /*firstLaneOnly=*/false);
    return success();
  }
  if (detached) {
    // A scalar tt.atomic_rmw executes once per block; each detached caller
    // publishes its own epoch.
    triton::ElementwiseInlineAsmOp::create(
        builder, loc, TypeRange{builder.getI64Type()},
        "atom.release.sys.global.exch.b64 $0, [$1], $2;",
        "=l,l,l", /*isPure=*/false, /*pack=*/1,
        ValueRange{destination->pointer, value});
    return success();
  }
  triton::AtomicRMWOp::create(
      builder, loc, builder.getI64Type(), triton::RMWOp::XCHG,
      destination->pointer, value, /*mask=*/nullptr,
      triton::MemSemantic::RELEASE,
      triton::MemSyncScope::SYSTEM);
  return success();
}

// Register values written straight to the peer: the payload packs into 32-bit
// little-endian words exactly as an mlx5 inline segment does, and a tail that
// is not a whole word is written with narrower stores so no neighbouring byte
// is touched. The destination must be four-byte aligned.
LogicalResult lowerPostInline(Operation *insertionPoint, PostInlineOp op,
                              bool detached) {
  OpBuilder builder(insertionPoint);
  Location loc = op.getLoc();
  auto destination = materializeRemoteRegion(op.getDestination(), op);
  if (failed(destination))
    return failure();
  const std::uint64_t bytes = op.getPayloadBytes();
  SmallVector<Value> words = packInlinePayload(
      builder, loc, op.getValues(), static_cast<std::uint32_t>((bytes + 3) / 4));
  auto conditional = scf::IfOp::create(builder, loc, TypeRange{},
                                       op.getPredicate(),
                                       /*withElseRegion=*/false);
  builder.setInsertionPoint(conditional.thenYield());
  Value base = pointerAddress(builder, loc, destination->pointer);
  auto storeNarrow = [&](std::uint64_t offset, Value word, unsigned width) {
    Value address = addressOffset(builder, loc, base, offset);
    if (!detached) {
      Type type = builder.getIntegerType(width);
      triton::StoreOp::create(
          builder, loc, pointerFromAddress(builder, loc, address, type),
          arith::TruncIOp::create(builder, loc, type, word),
          triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
      return;
    }
    // A detached caller is its own issuer, where a scalar tt.store would
    // narrow to one thread of a block it does not own. PTX stores a byte from
    // a 16-bit register.
    triton::ElementwiseInlineAsmOp::create(
        builder, loc, TypeRange{builder.getI32Type()},
        "{ mov.u32 $0, 0; st.global.b" + std::to_string(width) +
            " [$1], $2; }",
        "=r,l,h", /*isPure=*/false, /*pack=*/1,
        ValueRange{address, arith::TruncIOp::create(
                                builder, loc, builder.getI16Type(), word)});
  };
  for (auto [index, word] : llvm::enumerate(words)) {
    const std::uint64_t offset = index * 4;
    const std::uint64_t remaining = bytes - offset;
    if (remaining >= 4) {
      store32(builder, loc, base, offset, word, detached);
      continue;
    }
    if (remaining >= 2) {
      storeNarrow(offset, word, 16);
      word = arith::ShRUIOp::create(builder, loc, word,
                                    i32Constant(builder, loc, 16));
    }
    if (remaining % 2)
      storeNarrow(offset + (remaining & 2), word, 8);
  }
  return success();
}

LogicalResult lowerWaitSignal(WaitSignalOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  auto signal = materializeLocalRegion(op.getSignal(), op);
  if (failed(signal))
    return failure();
  auto signalType =
      dyn_cast<triton::PointerType>(signal->pointer.getType());
  if (!signalType || !signalType.getPointeeType().isInteger(64))
    return op.emitError("P2P wait_signal requires an i64 signal");

  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});

  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value observed =
      emit::acquireLoad64(builder, loc, signal->pointer);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, observed, op.getExpected());
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});

  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  return success();
}

LogicalResult lowerSystemFence(SystemFenceOp) {
  // A phase fence exists to order payload writes ahead of a separately
  // published queue head. Over peer memory the payload is ordinary stores and
  // every signal is a release-scope-system exchange, so the signal already
  // carries that ordering and there is no wire to fence against.
  return success();
}

LogicalResult lowerPublishHead(PublishHeadOp) {
  // Publication rings an mlx5 doorbell for externally constructed WQEs. Peer
  // memory has no queue, so the transfers are already visible when the
  // signal's release store retires.
  return success();
}

} // namespace

void LowerP2P::runOnOperation() {
  if (!hipDeviceLibrary.empty()) {
    if (!detached && !copyMode.empty()) {
      getOperation().emitError("HIP copy-mode selection requires a detached fragment");
      return signalPassFailure();
    }
    if (detached)
      getOperation()->setAttr("aboleth.hip-detached", UnitAttr::get(&getContext()));
    getOperation()->setAttr("aboleth.hip-device-library",
                            StringAttr::get(&getContext(), hipDeviceLibrary));
  }
  const bool detachedFragment = detached;
  const std::string mode = copyMode;
  // Peer memory needs no queue: a submission is just its transfers, and
  // completion is established by the release store the signal already emits.
  auto lowerSubmission = [&](SubmitOp submission, PollOp) -> LogicalResult {
    if (submission.getExternalPublication())
      return submission.emitError(
          "P2P cannot delegate publication to an RDMA phase adapter");
    // A put ahead of the submission's last signal is published by that
    // signal's release (emitCooperativeFence).
    std::optional<std::size_t> lastSignal;
    if (failed(forEachTransfer(
            submission,
            [](PostPutOp, std::size_t) { return success(); },
            [&](PostSignalOp, std::size_t index) {
              lastSignal = index;
              return success();
            },
            /*onNop=*/{},
            [](PostInlineOp, std::size_t) { return success(); })))
      return failure();
    auto onPut = [&](PostPutOp put, std::size_t index) {
      return lowerPostPut(submission, put, detachedFragment, mode,
                          lastSignal && index < *lastSignal);
    };
    auto onSignal = [&](PostSignalOp signal, std::size_t) {
      return lowerPostSignal(submission, signal, detachedFragment);
    };
    auto onInline = [&](PostInlineOp put, std::size_t) {
      return lowerPostInline(submission, put, detachedFragment);
    };
    return forEachTransfer(submission, onPut, onSignal, /*onNop=*/{},
                           onInline);
  };

  auto lowerDrainOp = [&](DrainOp) -> LogicalResult {
    // A p2p put is complete when its stores retire, and the signal's
    // release-system exchange already publishes them: there is no queue to
    // drain, so `quiet` is vacuous — the same closed-form answer
    // lowerSystemFence and lowerPublishHead give.
    return success();
  };

  QueueLoweringHooks hooks;
  hooks.backendName = "P2P";
  hooks.requireUnusedTokens = true;
  hooks.lowerSubmission = lowerSubmission;
  hooks.lowerSystemFence = lowerSystemFence;
  hooks.lowerPublishHead = lowerPublishHead;
  hooks.lowerWaitSignal = lowerWaitSignal;
  hooks.lowerDrain = lowerDrainOp;
  if (failed(runQueueLowering(getOperation(), hooks)))
    signalPassFailure();
}

} // namespace mlir::aboleth
