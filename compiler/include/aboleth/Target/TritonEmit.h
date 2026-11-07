#pragma once

// Triton IR emission primitives shared by every backend lowering.
//
// These were duplicated verbatim across the direct-mlx5 and host-proxy
// lowerings. Backends differ in the commands they build, not in how they
// materialize a constant, read a descriptor word, or publish a release store,
// so those primitives live here and every backend emits identical IR for them.

#include <algorithm>
#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::aboleth::emit {

inline StringAttr hipDeviceLibrary(OpBuilder &builder) {
  Operation *operation = builder.getInsertionBlock()->getParentOp();
  while (operation) {
    if (auto path = operation->getAttrOfType<StringAttr>("aboleth.hip-device-library"))
      return path;
    operation = operation->getParentOp();
  }
  return {};
}

inline bool hipDetached(OpBuilder &builder) {
  Operation *operation = builder.getInsertionBlock()->getParentOp();
  while (operation) {
    if (operation->hasAttr("aboleth.hip-detached")) return true;
    operation = operation->getParentOp();
  }
  return false;
}

inline Value hipCall(OpBuilder &builder, Location loc, Type result,
                      StringRef symbol, ValueRange arguments) {
  return triton::ExternElementwiseOp::create(
      builder, loc, result, arguments, "aboleth_hip",
      hipDeviceLibrary(builder).getValue(), symbol, /*pure=*/false);
}

inline Value i32Constant(OpBuilder &builder, Location loc,
                         std::uint32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getI32IntegerAttr(static_cast<std::int32_t>(value)));
}

inline Value i64Constant(OpBuilder &builder, Location loc,
                         std::uint64_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI64Type(),
      builder.getI64IntegerAttr(static_cast<std::int64_t>(value)));
}

inline Value loadValue(OpBuilder &builder, Location loc, Value pointer,
                       bool isVolatile = false) {
  return triton::LoadOp::create(
      builder, loc, pointer, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL, isVolatile);
}

// A word that only threads of this GPU write (a producer index, a ready head)
// needs only a GPU-scope acquire; one a host thread or a peer writes needs the
// system scope.
inline Value acquireLoad64(OpBuilder &builder, Location loc, Value pointer,
                           bool gpuScope = false) {
  if (hipDeviceLibrary(builder)) {
    Value address = triton::PtrToIntOp::create(builder, loc, builder.getI64Type(), pointer);
    return hipCall(builder, loc, builder.getI64Type(), "aboleth_hip_acquire64", {address});
  }
  auto load = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI64Type()},
      gpuScope ? "ld.acquire.gpu.global.L1::no_allocate.u64 $0, [$1];"
               : "ld.acquire.sys.global.L1::no_allocate.u64 $0, [$1];",
      "=l,l", /*isPure=*/false, /*pack=*/1, ValueRange{pointer});
  return load.getResult()[0];
}

// Endpoint descriptors are arrays of i64 words; each backend documents its own
// layout in its ABI header.
inline Value endpointWord(OpBuilder &builder, Location loc, Value endpoint,
                          std::uint32_t index) {
  Value offset = i32Constant(builder, loc, index);
  Value pointer = triton::AddPtrOp::create(
      builder, loc, endpoint.getType(), endpoint, offset);
  return loadValue(builder, loc, pointer);
}

inline Value low32(OpBuilder &builder, Location loc, Value word) {
  return arith::TruncIOp::create(builder, loc, builder.getI32Type(), word);
}

inline Value high32(OpBuilder &builder, Location loc, Value word) {
  Value shifted = arith::ShRUIOp::create(
      builder, loc, word, i64Constant(builder, loc, 32));
  return arith::TruncIOp::create(builder, loc, builder.getI32Type(), shifted);
}

inline Value zeroExtend64(OpBuilder &builder, Location loc, Value value) {
  return arith::ExtUIOp::create(builder, loc, builder.getI64Type(), value);
}

inline Value pointerFromAddress(OpBuilder &builder, Location loc, Value address,
                                Type pointeeType) {
  Type pointerType = triton::getPointerType(pointeeType);
  return triton::IntToPtrOp::create(builder, loc, pointerType, address);
}

inline Value pointerAddress(OpBuilder &builder, Location loc, Value pointer) {
  return triton::PtrToIntOp::create(
      builder, loc, builder.getI64Type(), pointer);
}

inline Value addressOffset(OpBuilder &builder, Location loc, Value address,
                           std::uint64_t offset) {
  return arith::AddIOp::create(
      builder, loc, address, i64Constant(builder, loc, offset));
}

// One 64-bit word of an endpoint's completion state block.
inline Value stateWordPointer(OpBuilder &builder, Location loc,
                              Value stateAddress, std::uint32_t word) {
  return pointerFromAddress(
      builder, loc,
      addressOffset(builder, loc, stateAddress,
                    word * sizeof(std::uint64_t)),
      builder.getI64Type());
}

// Triton lowers a scalar tt.store once per block, from lane 0 — correct when
// every thread computed the same value, and wrong for a detached fragment
// whose calling threads each own their own effect (their WQE slots differ).
// `perThread = true` emits the store as unpredicated inline asm instead.
// The payload's 32-bit words in address order. Inline data is not byte-swapped:
// only segment headers are big-endian, so each value contributes the bytes of
// its own little-endian representation and a plain store32 of the packed word
// lands them at the right addresses.
inline SmallVector<Value> packInlinePayload(OpBuilder &builder, Location loc,
                                     ValueRange values,
                                     std::uint32_t wordCount) {
  struct Field {
    Value value;
    std::uint64_t offset;
    std::uint64_t bytes;
  };
  SmallVector<Field> fields;
  std::uint64_t cursor = 0;
  for (Value value : values) {
    const std::uint64_t bytes =
        cast<IntegerType>(value.getType()).getWidth() / 8;
    fields.push_back({value, cursor, bytes});
    cursor += bytes;
  }

  SmallVector<Value> words;
  words.reserve(wordCount);
  for (std::uint32_t index = 0; index < wordCount; ++index) {
    const std::uint64_t wordStart = index * 4;
    Value word;
    for (const Field &field : fields) {
      const std::uint64_t start = std::max(field.offset, wordStart);
      const std::uint64_t stop =
          std::min(field.offset + field.bytes, wordStart + 4);
      if (start >= stop)
        continue;
      const std::uint64_t shiftRight = (start - field.offset) * 8;
      const std::uint64_t shiftLeft = (start - wordStart) * 8;
      const std::uint64_t chunkBits = (stop - start) * 8;
      const unsigned width =
          cast<IntegerType>(field.value.getType()).getWidth();
      Value piece = field.value;
      if (width == 64) {
        if (shiftRight)
          piece = arith::ShRUIOp::create(
              builder, loc, piece, i64Constant(builder, loc, shiftRight));
        piece = arith::TruncIOp::create(
            builder, loc, builder.getI32Type(), piece);
      } else {
        if (width < 32)
          piece = arith::ExtUIOp::create(
              builder, loc, builder.getI32Type(), piece);
        if (shiftRight)
          piece = arith::ShRUIOp::create(
              builder, loc, piece, i32Constant(builder, loc, shiftRight));
      }
      // Only a value that straddles this word's high edge needs masking; the
      // truncation and the zero extension already clear everything else.
      if (chunkBits < 32 && width - shiftRight > chunkBits)
        piece = arith::AndIOp::create(
            builder, loc, piece,
            i32Constant(builder, loc, (1u << chunkBits) - 1));
      if (shiftLeft)
        piece = arith::ShLIOp::create(
            builder, loc, piece, i32Constant(builder, loc, shiftLeft));
      word = word ? arith::OrIOp::create(builder, loc, word, piece) : piece;
    }
    words.push_back(word ? word : i32Constant(builder, loc, 0));
  }
  return words;
}

inline void store32(OpBuilder &builder, Location loc, Value address,
                    std::uint64_t offset, Value value,
                    bool perThread = false) {
  if (hipDeviceLibrary(builder) && (perThread || hipDetached(builder))) {
    Value target = addressOffset(builder, loc, address, offset);
    hipCall(builder, loc, builder.getI32Type(), "aboleth_hip_release32",
            {target, value, i32Constant(builder, loc, 0)});
    return;
  }
  if (perThread) {
    Value target = addressOffset(builder, loc, address, offset);
    triton::ElementwiseInlineAsmOp::create(
        builder, loc, TypeRange{value.getType()},
        "{ mov.b32 $0, $2; st.global.L1::no_allocate.b32 [$1], $2; }",
        "=r,l,r", /*isPure=*/false, /*pack=*/1, ValueRange{target, value});
    return;
  }
  Value pointer = pointerFromAddress(
      builder, loc, addressOffset(builder, loc, address, offset),
      builder.getI32Type());
  triton::StoreOp::create(
      builder, loc, pointer, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);
}

inline void store64(OpBuilder &builder, Location loc, Value address,
                    std::uint64_t offset, Value value,
                    bool perThread = false) {
  if (hipDeviceLibrary(builder) && (perThread || hipDetached(builder))) {
    Value target = addressOffset(builder, loc, address, offset);
    hipCall(builder, loc, builder.getI64Type(), "aboleth_hip_release64",
            {target, value, i32Constant(builder, loc, 0)});
    return;
  }
  if (perThread) {
    Value target = addressOffset(builder, loc, address, offset);
    triton::ElementwiseInlineAsmOp::create(
        builder, loc, TypeRange{value.getType()},
        "{ mov.b64 $0, $2; st.global.L1::no_allocate.b64 [$1], $2; }",
        "=l,l,l", /*isPure=*/false, /*pack=*/1, ValueRange{target, value});
    return;
  }
  Value pointer = pointerFromAddress(
      builder, loc, addressOffset(builder, loc, address, offset),
      builder.getI64Type());
  triton::StoreOp::create(
      builder, loc, pointer, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);
}

// The same choice for a store through an already-typed pointer.
inline void storeThrough64(OpBuilder &builder, Location loc, Value pointer,
                           Value value, bool perThread = false) {
  if (perThread) {
    triton::ElementwiseInlineAsmOp::create(
        builder, loc, TypeRange{value.getType()},
        "{ mov.b64 $0, $2; st.global.L1::no_allocate.b64 [$1], $2; }",
        "=l,l,l", /*isPure=*/false, /*pack=*/1, ValueRange{pointer, value});
    return;
  }
  triton::StoreOp::create(
      builder, loc, pointer, value, triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);
}

// The ordering a publication store carries. Exactly one store in a posting
// sequence needs system-release semantics — the doorbell/notification word an
// external observer (the NIC, or the staged host agent) watches; its release
// makes everything written before it visible along the causal chain. Words
// only other GPU threads consume (ready/completed state) publish with GPU
// scope, and words a later system release already covers (the DBREC) need no
// ordering of their own. This is the "do less than the static libraries"
// rule: NVSHMEM/DeepEP publish with st.release.gpu and relaxed segment
// stores; GDAKI's DBREC update is a relaxed system store.
enum class StoreOrder { ReleaseSys, ReleaseGpu, RelaxedSys, RelaxedGpu };

inline const char *storeOrderToken(StoreOrder order) {
  switch (order) {
  case StoreOrder::ReleaseSys:
    return "st.release.sys.global.L1::no_allocate";
  case StoreOrder::ReleaseGpu:
    return "st.release.gpu.global.L1::no_allocate";
  case StoreOrder::RelaxedSys:
    return "st.relaxed.sys.global.L1::no_allocate";
  case StoreOrder::RelaxedGpu:
    return "st.relaxed.gpu.global.L1::no_allocate";
  }
  return "st.release.sys.global.L1::no_allocate";
}

// One thread of the block performs the publishing store, so a batch is
// announced once no matter how many threads executed the emitting code.
// `firstLaneOnly = false` drops that implicit predicate: a detached device
// function's issuer is each calling thread (its entry wrapper has already
// narrowed a CTA-scoped issuer to one lane), and the caller's %tid.x says
// nothing about who must publish.
inline void releaseStore32(OpBuilder &builder, Location loc, Value pointer,
                           Value value, bool firstLaneOnly = true,
                           StoreOrder order = StoreOrder::ReleaseSys) {
  if (hipDeviceLibrary(builder)) {
    firstLaneOnly = firstLaneOnly && !hipDetached(builder);
    Value address = pointerAddress(builder, loc, pointer);
    hipCall(builder, loc, builder.getI32Type(), "aboleth_hip_release32",
            {address, value, i32Constant(builder, loc, firstLaneOnly)});
    return;
  }
  const std::string store = storeOrderToken(order);
  triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{value.getType()},
      firstLaneOnly
          ? "{ .reg .pred first; .reg .u32 tid; "
            "mov.u32 tid, %tid.x; setp.eq.u32 first, tid, 0; "
            "mov.b32 $0, $2; "
            "@first " + store + ".b32 [$1], $2; }"
          : "{ mov.b32 $0, $2; " + store + ".b32 [$1], $2; }",
      "=r,l,r", /*isPure=*/false, /*pack=*/1,
      ValueRange{pointer, value});
}

inline void releaseStore64(OpBuilder &builder, Location loc, Value pointer,
                           Value value, bool firstLaneOnly = true,
                           StoreOrder order = StoreOrder::ReleaseSys) {
  if (hipDeviceLibrary(builder)) {
    firstLaneOnly = firstLaneOnly && !hipDetached(builder);
    Value address = pointerAddress(builder, loc, pointer);
    hipCall(builder, loc, builder.getI64Type(), "aboleth_hip_release64",
            {address, value, i32Constant(builder, loc, firstLaneOnly)});
    return;
  }
  const std::string store = storeOrderToken(order);
  triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{value.getType()},
      firstLaneOnly
          ? "{ .reg .pred first; .reg .u32 tid; "
            "mov.u32 tid, %tid.x; setp.eq.u32 first, tid, 0; "
            "mov.b64 $0, $2; "
            "@first " + store + ".b64 [$1], $2; }"
          : "{ mov.b64 $0, $2; " + store + ".b64 [$1], $2; }",
      "=l,l,l", /*isPure=*/false, /*pack=*/1,
      ValueRange{pointer, value});
}

// A per-thread atomic add on a 64-bit word, returning the previous value in a
// register. `tt.atomic_rmw` on a scalar pointer executes once per block and
// Triton broadcasts its live result through shared memory; a detached device
// function has no shared memory to offer its caller, and its issuer contract
// is per-calling-thread anyway, so each thread performs its own reservation.
inline Value atomicAdd64(OpBuilder &builder, Location loc, Value pointer,
                         Value amount, bool ctaScope = false) {
  auto atom = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI64Type()},
      ctaScope ? "atom.relaxed.cta.global.add.u64 $0, [$1], $2;"
               : "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;",
      "=l,l,l", /*isPure=*/false, /*pack=*/1, ValueRange{pointer, amount});
  return atom.getResult()[0];
}

// %tid.x. Triton's data-parallel model exposes no thread identity, so a
// lowering that must name one lane reads the register directly.
inline Value readThreadIdX(OpBuilder &builder, Location loc) {
  auto tid = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI32Type()},
      "mov.u32 $0, %tid.x;", "=r", /*isPure=*/false, /*pack=*/1,
      ValueRange{});
  return tid.getResult()[0];
}

// True on the block's first lane. Descriptor construction and doorbells are
// block-wide effects that exactly one lane must perform; the inline-asm stores
// above carry their own `@first` predicate, and a call into a device library
// has to be guarded by this instead.
inline Value isFirstThread(OpBuilder &builder, Location loc) {
  return arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq,
      readThreadIdX(builder, loc), i32Constant(builder, loc, 0));
}

// %ctaid.x, %ctaid.y, %ctaid.z. Triton's data-parallel model exposes no block
// identity either (tt.get_program_id names a position in the program grid,
// which a launch may remap), so a lowering that must name one block reads the
// registers directly, exactly as readThreadIdX does for lanes.
inline Value readCtaIdX(OpBuilder &builder, Location loc) {
  auto ctaid = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI32Type()},
      "mov.u32 $0, %ctaid.x;", "=r", /*isPure=*/false, /*pack=*/1,
      ValueRange{});
  return ctaid.getResult()[0];
}

inline Value readCtaIdY(OpBuilder &builder, Location loc) {
  auto ctaid = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI32Type()},
      "mov.u32 $0, %ctaid.y;", "=r", /*isPure=*/false, /*pack=*/1,
      ValueRange{});
  return ctaid.getResult()[0];
}

inline Value readCtaIdZ(OpBuilder &builder, Location loc) {
  auto ctaid = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI32Type()},
      "mov.u32 $0, %ctaid.z;", "=r", /*isPure=*/false, /*pack=*/1,
      ValueRange{});
  return ctaid.getResult()[0];
}

// True on the grid's first block. Work that must happen once per kernel
// launch, rather than once per block, is guarded by this: `isFirstThread`
// narrows an effect to one lane of a block, and every block still runs it.
// All three axes are read, so a two- or three-dimensional launch still names
// exactly one block.
inline Value isFirstCta(OpBuilder &builder, Location loc) {
  Value merged = arith::OrIOp::create(
      builder, loc,
      arith::OrIOp::create(builder, loc, readCtaIdX(builder, loc),
                           readCtaIdY(builder, loc)),
      readCtaIdZ(builder, loc));
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, merged,
                               i32Constant(builder, loc, 0));
}

inline Value readClock64(OpBuilder &builder, Location loc) {
  if (hipDeviceLibrary(builder))
    return hipCall(builder, loc, builder.getI64Type(), "aboleth_hip_clock", {});
  auto clock = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI64Type()},
      "mov.u64 $0, %clock64;", "=l", /*isPure=*/false, /*pack=*/1,
      ValueRange{});
  return clock.getResult()[0];
}

// Spin until the observed word equals `expected`. Used to serialize batch
// publication into reservation order.
inline void waitUntilEqual(OpBuilder &builder, Location loc, Value pointer,
                           Value expected) {
  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value observed =
      loadValue(builder, loc, pointer, /*isVolatile=*/true);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, observed, expected);
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);
}

// Spin until the retired head leaves room for a batch ending at `batchEnd`.
// `completedPointer` addresses the endpoint's retired-head word and `capacity`
// is the queue depth in commands.
inline void waitForQueueCapacity(OpBuilder &builder, Location loc,
                                 Value completedPointer, Value batchEnd,
                                 Value capacityCount) {
  Value capacity = zeroExtend64(builder, loc, capacityCount);
  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value completedHead =
      loadValue(builder, loc, completedPointer, /*isVolatile=*/true);
  Value availableEnd =
      arith::AddIOp::create(builder, loc, completedHead, capacity);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, batchEnd, availableEnd);
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);
}

} // namespace mlir::aboleth::emit
