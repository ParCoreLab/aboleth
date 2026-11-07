#include "aboleth/Target/Passes.h"
#include "aboleth/Target/QueueLowering.h"
#include "aboleth/Target/TritonBridge.h"
#include "aboleth/Target/TritonEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "aboleth/Target/RdmaAbi.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <string>

namespace mlir::aboleth {

#define GEN_PASS_DEF_ABOLETHLOWERRDMAPASS
#include "aboleth/Target/Passes.h.inc"

namespace {

constexpr std::uint32_t kOpcodeRdmaWrite = 0x08;
constexpr std::uint32_t kOpcodeAtomicCompareSwap = 0x11;
constexpr std::uint32_t kWqeDataSegments = 3;
constexpr std::uint32_t kAtomicWqeDataSegments = 4;
constexpr std::uint32_t kCqUpdateWord = 0x08000000;
constexpr std::uint32_t kInlineSegment = 0x80000000;
// A WQE occupies whole 16-byte segments and whole 64-byte WQEBBs; the send
// queue's slots and its producer index are counted in WQEBBs.
constexpr std::uint64_t kWqeSegmentBytes = 16;
constexpr std::uint64_t kWqeBlockBytes = 64;
// ctrl + remote-address segments, then the inline-data segment header.
constexpr std::uint64_t kInlineHeaderOffset = 32;
constexpr std::uint64_t kInlinePayloadOffset = 36;
// fm = 3 in fm_ce_se, the kernel driver's MLX5_FENCE_MODE_STRONG_ORDERING
// (rdma-core's MLX5_WQE_CTRL_FENCE is fm = 4, 0x80). Publication planning
// selects it for a signal only where every receiver registration is strictly
// ordered; with relaxed-ordering registrations the signal is a responder
// compare-and-swap instead.
// TODO: confirm against the PRM that fm = 3 holds the write until every
// earlier WQE has completed at the responder.
constexpr std::uint32_t kStrongOrderWord = 0x60000000;

// The same geometry the ABI header states for the verifiers and transforms
// that refuse an oversized payload; keeping both definitions is only safe
// while they agree.
static_assert(kWqeSegmentBytes == ::aboleth::abi::kWqeSegmentBytes);
static_assert(kWqeBlockBytes == ::aboleth::abi::kWqeBlockBytes);
static_assert(kInlinePayloadOffset - kInlineHeaderOffset ==
              ::aboleth::abi::kInlineSegmentHeaderBytes);

struct DeviceLibrary {
  std::string path;

  // Building descriptors and ringing a doorbell are once-per-block effects, so
  // exactly one lane may enter the library. The direct emission reaches the
  // same shape through the lane predicate its publishing stores carry; a call
  // site has no such predicate and would otherwise post from all 32 lanes.
  Value call(OpBuilder &builder, Location loc, Value lane, StringRef symbol,
             ValueRange arguments) const {
    auto guard = scf::IfOp::create(
        builder, loc, TypeRange{builder.getI64Type()}, lane,
        /*withElseRegion=*/true);
    OpBuilder posting(guard.thenBlock(), guard.thenBlock()->begin());
    auto op = triton::ExternElementwiseOp::create(
        posting, loc, posting.getI64Type(), arguments,
        ::aboleth::abi::kRdmaDeviceLibraryName,
        path, symbol, /*pure=*/false);
    scf::YieldOp::create(posting, loc, ValueRange{op.getResult()});
    OpBuilder idle(guard.elseBlock(), guard.elseBlock()->begin());
    scf::YieldOp::create(
        idle, loc, ValueRange{emit::i64Constant(idle, loc, 0)});
    return guard.getResult(0);
  }
};

struct LowerRDMA : public impl::AbolethLowerRDMAPassBase<LowerRDMA> {
  using Base::Base;

  void runOnOperation() override;
};

using emit::acquireLoad64;
using emit::addressOffset;
using emit::high32;
using emit::i32Constant;
using emit::i64Constant;
using emit::isFirstThread;
using emit::loadValue;
using emit::packInlinePayload;
using emit::low32;
using emit::pointerAddress;
using emit::pointerFromAddress;
using emit::readClock64;
using emit::releaseStore32;
using emit::releaseStore64;
using emit::store32;
using emit::store64;
using emit::waitUntilEqual;
using emit::zeroExtend64;

// How descriptor words are read and RDMA-write WQEs are written, from the
// pass options; runOnOperation sets it for the duration of one run.
struct EmissionStyle {
  bool invariantDescriptors = false;
  bool vectorWqeStores = false;
};
thread_local EmissionStyle emission;

// A word of the endpoint descriptor. The host writes the descriptor once,
// when it creates the endpoint (aboleth_endpoint_rc.cu, _devx.cu), so every
// word is invariant while a kernel runs. With invariant-descriptors it is a
// pure non-coherent load, which CSE shares across every WQE of a batch; a
// plain load is re-issued per WQE, the queue stores between them being
// possible aliases.
Value endpointWord(OpBuilder &builder, Location loc, Value endpoint,
                   std::uint32_t index) {
  if (!emission.invariantDescriptors)
    return emit::endpointWord(builder, loc, endpoint, index);
  Value pointer = triton::AddPtrOp::create(
      builder, loc, endpoint.getType(), endpoint,
      i32Constant(builder, loc, index));
  auto load = triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI64Type()},
      "ld.global.nc.b64 $0, [$1];", "=l,l", /*isPure=*/true, /*pack=*/1,
      ValueRange{pointer});
  return load.getResult()[0];
}

// Sixteen bytes of a WQE in one store, from the issuing thread: the first
// thread of the block in a kernel (as Triton predicates a scalar tt.store),
// every caller of a detached fragment. WQE slots are 64-byte aligned.
void storeWqeSegment(OpBuilder &builder, Location loc, Value address,
                     std::uint64_t offset, Value word0, Value word1,
                     Value word2, Value word3, bool detached) {
  Value target = addressOffset(builder, loc, address, offset);
  triton::ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{builder.getI32Type()},
      detached
          ? "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.b32 $0, 0; }"
          : "{ .reg .pred first; .reg .b32 lane; mov.u32 lane, %tid.x; "
            "setp.eq.u32 first, lane, 0; "
            "@first st.global.v4.b32 [$1], {$2, $3, $4, $5}; "
            "mov.b32 $0, 0; }",
      "=r,l,r,r,r,r", /*isPure=*/false, /*pack=*/1,
      ValueRange{target, word0, word1, word2, word3});
}

// Ring position within a queue of `count` entries. Both runtime paths create
// send and completion queues with a power-of-two entry count (libmlx5 rejects
// anything else and the DevX path rounds up), so the wrap is a mask rather
// than a division.
Value ringSlot(OpBuilder &builder, Location loc, Value position,
               Value count64) {
  Value mask = arith::SubIOp::create(
      builder, loc, count64, i64Constant(builder, loc, 1));
  return arith::AndIOp::create(builder, loc, position, mask);
}

// Address of the WQEBB at producer position `position`.
Value sendQueueSlot(OpBuilder &builder, Location loc, Value sendQueue,
                    Value sendStride, Value sendWqeCount, Value position) {
  Value slot = ringSlot(builder, loc, position,
                        zeroExtend64(builder, loc, sendWqeCount));
  Value byteOffset = arith::MulIOp::create(
      builder, loc, slot, zeroExtend64(builder, loc, sendStride));
  return arith::AddIOp::create(builder, loc, sendQueue, byteOffset);
}

Value byteSwap32(OpBuilder &builder, Location loc, Value value) {
  Value byte0 = arith::AndIOp::create(
      builder, loc, value, i32Constant(builder, loc, 0x000000ff));
  Value byte1 = arith::AndIOp::create(
      builder, loc, value, i32Constant(builder, loc, 0x0000ff00));
  Value byte2 = arith::AndIOp::create(
      builder, loc, value, i32Constant(builder, loc, 0x00ff0000));
  Value byte3 = arith::AndIOp::create(
      builder, loc, value, i32Constant(builder, loc, 0xff000000));
  byte0 = arith::ShLIOp::create(
      builder, loc, byte0, i32Constant(builder, loc, 24));
  byte1 = arith::ShLIOp::create(
      builder, loc, byte1, i32Constant(builder, loc, 8));
  byte2 = arith::ShRUIOp::create(
      builder, loc, byte2, i32Constant(builder, loc, 8));
  byte3 = arith::ShRUIOp::create(
      builder, loc, byte3, i32Constant(builder, loc, 24));
  return arith::OrIOp::create(
      builder, loc,
      arith::OrIOp::create(builder, loc, byte0, byte1),
      arith::OrIOp::create(builder, loc, byte2, byte3));
}

Value byteSwap64(OpBuilder &builder, Location loc, Value value) {
  Value low = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), value);
  Value shifted = arith::ShRUIOp::create(
      builder, loc, value, i64Constant(builder, loc, 32));
  Value high = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), shifted);
  Value swappedLow =
      zeroExtend64(builder, loc, byteSwap32(builder, loc, low));
  Value swappedHigh =
      zeroExtend64(builder, loc, byteSwap32(builder, loc, high));
  swappedLow = arith::ShLIOp::create(
      builder, loc, swappedLow, i64Constant(builder, loc, 32));
  return arith::OrIOp::create(builder, loc, swappedLow, swappedHigh);
}

// Bytes the inline-data segment occupies: its 4-byte header plus the payload,
// rounded up to a whole 16-byte segment.
std::uint64_t inlineSegmentBytes(std::uint64_t payloadBytes) {
  const std::uint64_t used =
      kInlinePayloadOffset - kInlineHeaderOffset + payloadBytes;
  return ((used + kWqeSegmentBytes - 1) / kWqeSegmentBytes) *
         kWqeSegmentBytes;
}

// The ds field of an inline write: control, remote address, and inline data.
std::uint32_t inlineWqeSegments(std::uint64_t payloadBytes) {
  return static_cast<std::uint32_t>(
      2 + inlineSegmentBytes(payloadBytes) / kWqeSegmentBytes);
}

// What the command costs the producer index. Every other command this lowering
// builds fits one WQEBB; an inline write is the only one that may not.
std::uint32_t inlineWqeBlocks(std::uint64_t payloadBytes) {
  return static_cast<std::uint32_t>(
      ::aboleth::abi::inline_wqe_blocks(payloadBytes));
}

// A predicate this emission can prove is always active. MLIR's `m_One` asks
// whether the defining operation carries the ConstantLike trait, and a trait's
// TypeID is not shared across the plugin boundary, so the plugin matches the
// constant operation itself instead.
bool isAlwaysActive(Value predicate) {
  auto constant = predicate.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto boolean = dyn_cast<BoolAttr>(constant.getValue());
  return boolean && boolean.getValue();
}

std::uint64_t inlinePayloadBytes(ValueRange values) {
  std::uint64_t bytes = 0;
  for (Value value : values)
    bytes += cast<IntegerType>(value.getType()).getWidth() / 8;
  return bytes;
}


Value byteSwap16To32(OpBuilder &builder, Location loc, Value value) {
  Value extended =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), value);
  Value low = arith::AndIOp::create(
      builder, loc, extended, i32Constant(builder, loc, 0xff));
  Value high = arith::AndIOp::create(
      builder, loc,
      arith::ShRUIOp::create(
          builder, loc, extended, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, 0xff));
  low = arith::ShLIOp::create(
      builder, loc, low, i32Constant(builder, loc, 8));
  return arith::OrIOp::create(builder, loc, low, high);
}

// The RDMA endpoint descriptor keeps the address of its completion state block
// in word 6.
Value completionStatePointer(OpBuilder &builder, Location loc, Value endpoint,
                             std::uint32_t word) {
  Value stateAddress = endpointWord(builder, loc, endpoint, 6);
  return emit::stateWordPointer(builder, loc, stateAddress, word);
}

void waitForQueueCapacity(OpBuilder &builder, Location loc, Value endpoint,
                          Value batchEnd, Value sendWqeCount) {
  Value completed =
      completionStatePointer(builder, loc, endpoint, /*word=*/1);
  emit::waitForQueueCapacity(
      builder, loc, completed, batchEnd, sendWqeCount);
}

LogicalResult validateEndpoint(Value endpoint, Operation *operation) {
  auto pointerType = dyn_cast<triton::PointerType>(endpoint.getType());
  if (!pointerType || !pointerType.getPointeeType().isInteger(64))
    return operation->emitError(
        "RDMA endpoint must be a pointer to the 96-byte i64 descriptor");
  return success();
}

// Word 5 of the descriptor holds the address of the 64-bit producer index.
Value producerIndexPointer(OpBuilder &builder, Location loc, Value endpoint) {
  Value stateAddress = endpointWord(builder, loc, endpoint, 5);
  return pointerFromAddress(
      builder, loc, stateAddress, builder.getI64Type());
}

Value reserveBatch(OpBuilder &builder, Location loc, Value endpoint,
                   std::uint32_t batchSize,
                   triton::MemSyncScope scope = triton::MemSyncScope::GPU,
                   bool detached = false, bool exclusive = false) {
  Value statePointer = producerIndexPointer(builder, loc, endpoint);
  // An exclusive endpoint is owned by a single issuing thread, so the
  // reservation needs no atomicity at all: a plain load/add/store pair in
  // registers. (Valid only under the exclusive-issuer contract the caller
  // opted into; concurrent producers on such an endpoint corrupt the queue.)
  if (exclusive) {
    Value previous = loadValue(builder, loc, statePointer);
    Value next = arith::AddIOp::create(
        builder, loc, previous, i64Constant(builder, loc, batchSize));
    emit::storeThrough64(builder, loc, statePointer, next, detached);
    return previous;
  }
  // In-kernel, the scalar atomic executes once per block and Triton
  // broadcasts its result to every thread — through shared memory, which a
  // detached fragment may not impose on its caller. Detached, each calling
  // thread is its own issuer and performs its own reservation in a register.
  if (detached)
    return emit::atomicAdd64(
        builder, loc, statePointer, i64Constant(builder, loc, batchSize),
        /*ctaScope=*/scope == triton::MemSyncScope::CTA);
  return triton::AtomicRMWOp::create(
      builder, loc, builder.getI64Type(), triton::RMWOp::ADD,
      statePointer, i64Constant(builder, loc, batchSize), /*mask=*/nullptr,
      triton::MemSemantic::RELAXED, scope);
}

LogicalResult writeWqe(Operation *insertionPoint, Location loc, Value endpoint,
                       Value destination, Value source, Value bytes,
                       Value sourceLkey, Value destinationRkey,
                       Value predicate, Value producerIndex,
                       Value requestCompletion,
                       std::uint32_t controlWord = 0,
                       bool detached = false) {
  if (failed(validateEndpoint(endpoint, insertionPoint)))
    return failure();
  OpBuilder builder(insertionPoint);

  Value sendQueue = endpointWord(builder, loc, endpoint, 0);
  Value layout = endpointWord(builder, loc, endpoint, 8);
  Value identifiers = endpointWord(builder, loc, endpoint, 9);

  Value sendStride = low32(builder, loc, layout);
  Value sendWqeCount = high32(builder, loc, layout);
  Value qpn = low32(builder, loc, identifiers);

  Value wqeAddress = sendQueueSlot(builder, loc, sendQueue, sendStride,
                                   sendWqeCount, producerIndex);

  Value producer32 = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), producerIndex);
  producer32 = arith::AndIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 0xffff));
  Value rdmaOpcodeAndIndex = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, producer32, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, kOpcodeRdmaWrite));
  Value nopOpcodeAndIndex = arith::ShLIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 8));
  Value opcodeAndIndex = arith::SelectOp::create(
      builder, loc, predicate, rdmaOpcodeAndIndex, nopOpcodeAndIndex);
  opcodeAndIndex = byteSwap32(builder, loc, opcodeAndIndex);

  Value rdmaQpnAndSegments = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, qpn, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, kWqeDataSegments));
  Value nopQpnAndSegments = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, qpn, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, 1));
  Value qpnAndSegments = arith::SelectOp::create(
      builder, loc, predicate, rdmaQpnAndSegments, nopQpnAndSegments);
  qpnAndSegments = byteSwap32(builder, loc, qpnAndSegments);

  Value destinationAddress =
      byteSwap64(builder, loc, pointerAddress(builder, loc, destination));
  Value sourceAddress =
      byteSwap64(builder, loc, pointerAddress(builder, loc, source));

  Value completionWord = arith::SelectOp::create(
      builder, loc, requestCompletion,
      i32Constant(builder, loc, kCqUpdateWord | controlWord),
      i32Constant(builder, loc, controlWord));
  if (emission.vectorWqeStores) {
    Value zero = i32Constant(builder, loc, 0);
    // Control, remote-address and data segments: three 16-byte stores where
    // the word-by-word layout takes ten. The 64-bit addresses are already
    // big-endian values; their little-endian halves land in memory order.
    storeWqeSegment(builder, loc, wqeAddress, 0, opcodeAndIndex,
                    qpnAndSegments, completionWord, zero, detached);
    storeWqeSegment(builder, loc, wqeAddress, 16,
                    low32(builder, loc, destinationAddress),
                    high32(builder, loc, destinationAddress),
                    byteSwap32(builder, loc, destinationRkey), zero, detached);
    storeWqeSegment(builder, loc, wqeAddress, 32,
                    byteSwap32(builder, loc, bytes),
                    byteSwap32(builder, loc, sourceLkey),
                    low32(builder, loc, sourceAddress),
                    high32(builder, loc, sourceAddress), detached);
    return success();
  }
  store32(builder, loc, wqeAddress, 0, opcodeAndIndex, detached);
  store32(builder, loc, wqeAddress, 4, qpnAndSegments, detached);
  store32(builder, loc, wqeAddress, 8, completionWord, detached);
  store32(builder, loc, wqeAddress, 12, i32Constant(builder, loc, 0),
          detached);
  store64(builder, loc, wqeAddress, 16, destinationAddress, detached);
  store32(builder, loc, wqeAddress, 24,
          byteSwap32(builder, loc, destinationRkey), detached);
  store32(builder, loc, wqeAddress, 28, i32Constant(builder, loc, 0),
          detached);
  store32(builder, loc, wqeAddress, 32,
          byteSwap32(builder, loc, bytes), detached);
  store32(builder, loc, wqeAddress, 36,
          byteSwap32(builder, loc, sourceLkey), detached);
  store64(builder, loc, wqeAddress, 40, sourceAddress, detached);
  return success();
}

LogicalResult writeInlineSignalWqe(
    Operation *insertionPoint, Location loc, Value endpoint,
    Value destination, Value signalValue, Value destinationRkey,
    Value producerIndex, Value requestCompletion,
    std::uint32_t controlWord = 0, bool detached = false,
    Value predicate = {}) {
  if (failed(validateEndpoint(endpoint, insertionPoint)))
    return failure();

  OpBuilder builder(insertionPoint);
  Value sendQueue = endpointWord(builder, loc, endpoint, 0);
  Value layout = endpointWord(builder, loc, endpoint, 8);
  Value identifiers = endpointWord(builder, loc, endpoint, 9);
  Value sendStride = low32(builder, loc, layout);
  Value sendWqeCount = high32(builder, loc, layout);
  Value qpn = low32(builder, loc, identifiers);

  Value wqeAddress = sendQueueSlot(builder, loc, sendQueue, sendStride,
                                   sendWqeCount, producerIndex);

  Value producer32 = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), producerIndex);
  producer32 = arith::AndIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 0xffff));
  Value opcodeAndIndex = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, producer32, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, kOpcodeRdmaWrite));
  // An elided signal keeps its slot as a no-op of the same size that neither
  // writes nor fences: it publishes nothing and orders nothing.
  Value control = i32Constant(builder, loc, controlWord);
  if (predicate) {
    opcodeAndIndex = arith::SelectOp::create(
        builder, loc, predicate, opcodeAndIndex,
        arith::ShLIOp::create(builder, loc, producer32,
                              i32Constant(builder, loc, 8)));
    control = arith::SelectOp::create(builder, loc, predicate, control,
                                      i32Constant(builder, loc, 0));
  }
  opcodeAndIndex = byteSwap32(builder, loc, opcodeAndIndex);
  Value qpnAndSegments = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, qpn, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, kWqeDataSegments));
  qpnAndSegments = byteSwap32(builder, loc, qpnAndSegments);
  Value completionWord = arith::SelectOp::create(
      builder, loc, requestCompletion,
      arith::OrIOp::create(builder, loc, control,
                           i32Constant(builder, loc, kCqUpdateWord)),
      control);
  Value destinationAddress =
      byteSwap64(builder, loc, pointerAddress(builder, loc, destination));
  Value signalLow = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), signalValue);
  Value signalHigh64 = arith::ShRUIOp::create(
      builder, loc, signalValue, i64Constant(builder, loc, 32));
  Value signalHigh = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), signalHigh64);

  store32(builder, loc, wqeAddress, 0, opcodeAndIndex, detached);
  store32(builder, loc, wqeAddress, 4, qpnAndSegments, detached);
  store32(builder, loc, wqeAddress, 8, completionWord, detached);
  store32(builder, loc, wqeAddress, 12, i32Constant(builder, loc, 0),
          detached);
  store64(builder, loc, wqeAddress, 16, destinationAddress, detached);
  store32(builder, loc, wqeAddress, 24,
          byteSwap32(builder, loc, destinationRkey), detached);
  store32(builder, loc, wqeAddress, 28, i32Constant(builder, loc, 0),
          detached);
  store32(
      builder, loc, wqeAddress, 32,
      byteSwap32(
          builder, loc,
          i32Constant(
              builder, loc,
              kInlineSegment | sizeof(std::uint64_t))),
      detached);
  store32(builder, loc, wqeAddress, 36, signalLow, detached);
  store32(builder, loc, wqeAddress, 40, signalHigh, detached);
  store32(builder, loc, wqeAddress, 44, i32Constant(builder, loc, 0),
          detached);
  return success();
}

// A payload the caller computed in registers, embedded in the command. There is
// no local region, no lkey, and nothing for the adapter to read back over PCIe.
LogicalResult writeInlinePayloadWqe(
    Operation *insertionPoint, Location loc, Value endpoint,
    Value destination, ValueRange values, Value destinationRkey,
    Value predicate, Value producerIndex, Value requestCompletion,
    bool detached = false) {
  if (failed(validateEndpoint(endpoint, insertionPoint)))
    return failure();
  const std::uint64_t payloadBytes = inlinePayloadBytes(values);
  const std::uint32_t segments = inlineWqeSegments(payloadBytes);

  OpBuilder builder(insertionPoint);
  Value sendQueue = endpointWord(builder, loc, endpoint, 0);
  Value layout = endpointWord(builder, loc, endpoint, 8);
  Value identifiers = endpointWord(builder, loc, endpoint, 9);
  Value sendStride = low32(builder, loc, layout);
  Value sendWqeCount = high32(builder, loc, layout);
  Value qpn = low32(builder, loc, identifiers);

  // The send queue is circular in WQEBBs, so a WQE longer than one WQEBB
  // continues at the start of the buffer instead of running past its end. Each
  // block therefore resolves its own slot, exactly as mlx5's own providers wrap
  // a long WQE at the queue end.
  SmallVector<Value> blockBases;
  auto blockBase = [&](std::uint64_t block) -> Value {
    while (blockBases.size() <= block) {
      const std::uint64_t index = blockBases.size();
      Value position = producerIndex;
      if (index)
        position = arith::AddIOp::create(
            builder, loc, producerIndex, i64Constant(builder, loc, index));
      blockBases.push_back(sendQueueSlot(builder, loc, sendQueue, sendStride,
                                         sendWqeCount, position));
    }
    return blockBases[block];
  };
  auto storeWord = [&](std::uint64_t offset, Value value) {
    store32(builder, loc, blockBase(offset / kWqeBlockBytes),
            offset % kWqeBlockBytes, value, detached);
  };
  auto storeDouble = [&](std::uint64_t offset, Value value) {
    store64(builder, loc, blockBase(offset / kWqeBlockBytes),
            offset % kWqeBlockBytes, value, detached);
  };

  Value producer32 = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), producerIndex);
  producer32 = arith::AndIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 0xffff));
  Value shiftedIndex = arith::ShLIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 8));
  Value rdmaOpcodeAndIndex = arith::OrIOp::create(
      builder, loc, shiftedIndex,
      i32Constant(builder, loc, kOpcodeRdmaWrite));
  // An inactive predicate becomes an ordered NOP that keeps the write's segment
  // count, so the command consumes the same WQEBBs either way and the next WQE
  // still begins where the reservation says it does. Its completion request is
  // kept, so a disabled last command still retires the batch.
  Value opcodeAndIndex = arith::SelectOp::create(
      builder, loc, predicate, rdmaOpcodeAndIndex, shiftedIndex);
  opcodeAndIndex = byteSwap32(builder, loc, opcodeAndIndex);
  Value qpnAndSegments = byteSwap32(
      builder, loc,
      arith::OrIOp::create(
          builder, loc,
          arith::ShLIOp::create(
              builder, loc, qpn, i32Constant(builder, loc, 8)),
          i32Constant(builder, loc, segments)));
  Value completionWord = arith::SelectOp::create(
      builder, loc, requestCompletion,
      i32Constant(builder, loc, kCqUpdateWord), i32Constant(builder, loc, 0));
  Value destinationAddress =
      byteSwap64(builder, loc, pointerAddress(builder, loc, destination));

  storeWord(0, opcodeAndIndex);
  storeWord(4, qpnAndSegments);
  storeWord(8, completionWord);
  storeWord(12, i32Constant(builder, loc, 0));
  storeDouble(16, destinationAddress);
  storeWord(24, byteSwap32(builder, loc, destinationRkey));
  storeWord(28, i32Constant(builder, loc, 0));
  storeWord(kInlineHeaderOffset,
            byteSwap32(
                builder, loc,
                i32Constant(builder, loc,
                            kInlineSegment |
                                static_cast<std::uint32_t>(payloadBytes))));

  // Every word of the inline segment is written, so the padding the adapter
  // ignores is still deterministic rather than whatever the slot last held.
  const std::uint32_t wordCount = static_cast<std::uint32_t>(
      (inlineSegmentBytes(payloadBytes) -
       (kInlinePayloadOffset - kInlineHeaderOffset)) / 4);
  SmallVector<Value> words =
      packInlinePayload(builder, loc, values, wordCount);
  for (auto [index, word] : llvm::enumerate(words))
    storeWord(kInlinePayloadOffset + index * 4, word);
  return success();
}

// Strong publication unless planning selected a fenced write: a responder
// compare-and-swap whose read cannot pass the relaxed-ordering payload writes
// before it. It replaces only the value it expects, `epoch - 1` by default.
LogicalResult writeAtomicSignalWqe(
    Operation *insertionPoint, Location loc, Value endpoint,
    Value destination, Value signalValue, Value destinationRkey,
    Value producerIndex, Value requestCompletion,
    bool detached = false, Value expected = {}, Value predicate = {}) {
  if (failed(validateEndpoint(endpoint, insertionPoint)))
    return failure();

  OpBuilder builder(insertionPoint);
  Value sendQueue = endpointWord(builder, loc, endpoint, 0);
  Value layout = endpointWord(builder, loc, endpoint, 8);
  Value identifiers = endpointWord(builder, loc, endpoint, 9);
  Value atomicResultFields = endpointWord(builder, loc, endpoint, 11);
  Value sendStride = low32(builder, loc, layout);
  Value sendWqeCount = high32(builder, loc, layout);
  Value qpn = low32(builder, loc, identifiers);
  Value atomicResultLkey =
      low32(builder, loc, atomicResultFields);
  Value atomicResultAddress = addressOffset(
      builder, loc, endpointWord(builder, loc, endpoint, 6),
      ::aboleth::abi::kRdmaAtomicResultOffset);

  Value wqeAddress = sendQueueSlot(builder, loc, sendQueue, sendStride,
                                   sendWqeCount, producerIndex);

  Value producer32 = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), producerIndex);
  producer32 = arith::AndIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 0xffff));
  Value shiftedIndex = arith::ShLIOp::create(
      builder, loc, producer32, i32Constant(builder, loc, 8));
  Value opcodeAndIndex = arith::OrIOp::create(
      builder, loc, shiftedIndex,
      i32Constant(builder, loc, kOpcodeAtomicCompareSwap));
  // An elided signal keeps its slot as a no-op of the same size.
  if (predicate)
    opcodeAndIndex = arith::SelectOp::create(builder, loc, predicate,
                                             opcodeAndIndex, shiftedIndex);
  opcodeAndIndex = byteSwap32(builder, loc, opcodeAndIndex);
  Value qpnAndSegments = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, qpn, i32Constant(builder, loc, 8)),
      i32Constant(builder, loc, kAtomicWqeDataSegments));
  qpnAndSegments = byteSwap32(builder, loc, qpnAndSegments);
  Value completionWord = arith::SelectOp::create(
      builder, loc, requestCompletion,
      i32Constant(builder, loc, kCqUpdateWord),
      i32Constant(builder, loc, 0));

  Value predecessor =
      expected ? expected
               : arith::SubIOp::create(builder, loc, signalValue,
                                       i64Constant(builder, loc, 1));
  Value destinationAddress =
      byteSwap64(builder, loc, pointerAddress(builder, loc, destination));
  atomicResultAddress =
      byteSwap64(builder, loc, atomicResultAddress);

  store32(builder, loc, wqeAddress, 0, opcodeAndIndex, detached);
  store32(builder, loc, wqeAddress, 4, qpnAndSegments, detached);
  store32(builder, loc, wqeAddress, 8, completionWord, detached);
  store32(builder, loc, wqeAddress, 12, i32Constant(builder, loc, 0),
          detached);
  store64(builder, loc, wqeAddress, 16, destinationAddress, detached);
  store32(builder, loc, wqeAddress, 24,
          byteSwap32(builder, loc, destinationRkey), detached);
  store32(builder, loc, wqeAddress, 28, i32Constant(builder, loc, 0),
          detached);
  store64(builder, loc, wqeAddress, 32,
          byteSwap64(builder, loc, signalValue), detached);
  store64(builder, loc, wqeAddress, 40,
          byteSwap64(builder, loc, predecessor), detached);
  store32(builder, loc, wqeAddress, 48,
          byteSwap32(
              builder, loc,
              i32Constant(builder, loc, sizeof(std::uint64_t))),
          detached);
  store32(builder, loc, wqeAddress, 52,
          byteSwap32(builder, loc, atomicResultLkey), detached);
  store64(builder, loc, wqeAddress, 56, atomicResultAddress, detached);
  return success();
}

LogicalResult writeCompletionWqe(Operation *insertionPoint, Location loc,
                                 Value endpoint, Value producerIndex,
                                 bool detached = false) {
  OpBuilder builder(insertionPoint);
  Value zero = i32Constant(builder, loc, 0);
  Value predicate = arith::ConstantOp::create(
      builder, loc, builder.getI1Type(), builder.getBoolAttr(false));
  Value requestCompletion = arith::ConstantOp::create(
      builder, loc, builder.getI1Type(), builder.getBoolAttr(true));
  return writeWqe(
      insertionPoint, loc, endpoint,
      /*destination=*/endpoint, /*source=*/endpoint,
      /*bytes=*/zero, /*sourceLkey=*/zero, /*destinationRkey=*/zero,
      predicate, producerIndex, requestCompletion,
      /*controlWord=*/0, detached);
}

LogicalResult lowerSubmit(Operation *op, Location loc, Value endpoint,
                          Value head, bool detached = false,
                          emit::StoreOrder doorbellOrder =
                              emit::StoreOrder::ReleaseSys) {
  if (failed(validateEndpoint(endpoint, op)))
    return failure();
  OpBuilder builder(op);
  Value sendDbrecAddress = endpointWord(builder, loc, endpoint, 1);
  Value blueflameAddress = endpointWord(builder, loc, endpoint, 2);
  Value identifiers = endpointWord(builder, loc, endpoint, 9);
  Value qpn = low32(builder, loc, identifiers);

  Value head32 =
      arith::TruncIOp::create(builder, loc, builder.getI32Type(), head);
  Value head16 = arith::AndIOp::create(
      builder, loc, head32, i32Constant(builder, loc, 0xffff));

  Value dbrecValue =
      byteSwap32(builder, loc, head16);
  Value dbrecPointer = pointerFromAddress(
      builder, loc, sendDbrecAddress, builder.getI32Type());
  // The DBREC needs no ordering of its own: the doorbell's system release
  // below covers it for both observers (the NIC behind the BlueFlame write,
  // the staged agent behind the watched cell). The hand-written facade and
  // the Tier-2/3 adapter already publish it relaxed — a release here was
  // strictly stronger than either, for nothing.
  releaseStore32(builder, loc, dbrecPointer, dbrecValue,
                 /*firstLaneOnly=*/!detached,
                 emit::StoreOrder::RelaxedSys);

  Value doorbellIndex = arith::ShLIOp::create(
      builder, loc, head16, i32Constant(builder, loc, 8));
  doorbellIndex = byteSwap32(builder, loc, doorbellIndex);
  Value doorbellQpn = arith::ShLIOp::create(
      builder, loc, qpn, i32Constant(builder, loc, 8));
  doorbellQpn = byteSwap32(builder, loc, doorbellQpn);
  Value doorbell = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(
          builder, loc, zeroExtend64(builder, loc, doorbellQpn),
          i64Constant(builder, loc, 32)),
      zeroExtend64(builder, loc, doorbellIndex));
  // The release's scope is the caller's: GPU scope when every queue and the
  // payload are GPU-resident (the NIC reads them through L2, which the release
  // has reached before the ring), system scope otherwise.
  Value blueflamePointer = pointerFromAddress(
      builder, loc, blueflameAddress, builder.getI64Type());
  releaseStore64(builder, loc, blueflamePointer, doorbell,
                 /*firstLaneOnly=*/!detached, doorbellOrder);
  return success();
}

LogicalResult lowerCompletion(Operation *op, Location loc, Value endpoint,
                              Value batchBase, Value batchEnd,
                              Value predicate = {},
                              bool coalesced = false,
                              bool detached = false) {
  if (failed(validateEndpoint(endpoint, op)))
    return failure();
  OpBuilder builder(op);
  if (predicate) {
    auto conditional = scf::IfOp::create(
        builder, loc, TypeRange{}, predicate,
        /*withElseRegion=*/false);
    builder.setInsertionPoint(
        conditional.thenBlock()->getTerminator());
  }

  Value completionQueue = endpointWord(builder, loc, endpoint, 3);
  Value completionDbrecAddress = endpointWord(builder, loc, endpoint, 4);
  Value qpnAndCqeSize = endpointWord(builder, loc, endpoint, 9);
  Value countAndVersion = endpointWord(builder, loc, endpoint, 10);
  Value cqeSize = high32(builder, loc, qpnAndCqeSize);
  Value cqeCount = low32(builder, loc, countAndVersion);

  Value consumerState =
      completionStatePointer(builder, loc, endpoint, /*word=*/0);
  Value completedState =
      completionStatePointer(builder, loc, endpoint, /*word=*/1);
  Value completionBase = batchBase;
  if (coalesced)
    completionBase = loadValue(
        builder, loc, completedState, /*isVolatile=*/true);
  waitUntilEqual(builder, loc, completedState, completionBase);
  Value consumer = loadValue(builder, loc, consumerState);

  Value count64 = zeroExtend64(builder, loc, cqeCount);
  Value size64 = zeroExtend64(builder, loc, cqeSize);
  Value entry = ringSlot(builder, loc, consumer, count64);
  Value entryOffset = arith::MulIOp::create(builder, loc, entry, size64);
  Value is128 = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, cqeSize,
      i32Constant(builder, loc, 128));
  Value cqe64Offset = arith::SelectOp::create(
      builder, loc, is128, i64Constant(builder, loc, 64),
      i64Constant(builder, loc, 0));
  Value cqeAddress = arith::AddIOp::create(
      builder, loc, completionQueue,
      arith::AddIOp::create(builder, loc, entryOffset, cqe64Offset));

  Value opOwnPointer = pointerFromAddress(
      builder, loc, addressOffset(builder, loc, cqeAddress, 63),
      builder.getI8Type());
  Value counterPointer = pointerFromAddress(
      builder, loc, addressOffset(builder, loc, cqeAddress, 60),
      builder.getI16Type());
  // The owner bit flips every lap of the ring: bit log2(count) of the
  // consumer index, which for a power-of-two count is `consumer & count`.
  Value expectedOwner = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(),
      arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ne,
          arith::AndIOp::create(builder, loc, consumer, count64),
          i64Constant(builder, loc, 0)));
  Value targetCounter = arith::AndIOp::create(
      builder, loc,
      arith::SubIOp::create(
          builder, loc, batchEnd, i64Constant(builder, loc, 1)),
      i64Constant(builder, loc, 0xffff));
  Value timeoutCycles = endpointWord(builder, loc, endpoint, 7);
  Value pollStart = readClock64(builder, loc);

  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value opOwn = loadValue(builder, loc, opOwnPointer, /*isVolatile=*/true);
  Value opOwn64 = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), opOwn);
  Value opcode = arith::ShRUIOp::create(
      builder, loc, opOwn64, i64Constant(builder, loc, 4));
  Value owner = arith::AndIOp::create(
      builder, loc, opOwn64, i64Constant(builder, loc, 1));
  Value ownerReady = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, owner, expectedOwner);
  Value opcodeValid = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, opcode,
      i64Constant(builder, loc, 15));
  Value completionAvailable =
      arith::AndIOp::create(builder, loc, ownerReady, opcodeValid);
  Value now = readClock64(builder, loc);
  Value elapsed =
      arith::SubIOp::create(builder, loc, now, pollStart);
  Value timedOut = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, elapsed, timeoutCycles);
  Value done =
      arith::OrIOp::create(builder, loc, completionAvailable, timedOut);
  Value keepWaiting = arith::XOrIOp::create(
      builder, loc, done,
      arith::ConstantOp::create(
          builder, loc, builder.getI1Type(), builder.getBoolAttr(true)));
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);

  builder.setInsertionPointAfter(loop);
  Value finalOpOwn =
      loadValue(builder, loc, opOwnPointer, /*isVolatile=*/true);
  Value finalOpOwn64 = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), finalOpOwn);
  Value finalOpcode = arith::ShRUIOp::create(
      builder, loc, finalOpOwn64, i64Constant(builder, loc, 4));
  Value finalOwner = arith::AndIOp::create(
      builder, loc, finalOpOwn64, i64Constant(builder, loc, 1));
  Value finalCounter = loadValue(
      builder, loc, counterPointer, /*isVolatile=*/true);
  finalCounter = byteSwap16To32(builder, loc, finalCounter);
  Value finalCounter64 = zeroExtend64(builder, loc, finalCounter);
  Value opcodeReady = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, finalOpcode,
      i64Constant(builder, loc, 0));
  Value finalOwnerReady = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, finalOwner, expectedOwner);
  Value counterReady = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, finalCounter64, targetCounter);
  Value completionSuccess = arith::AndIOp::create(
      builder, loc, opcodeReady,
      arith::AndIOp::create(builder, loc, finalOwnerReady, counterReady));

  Value advancedConsumer = arith::AddIOp::create(
      builder, loc, consumer, i64Constant(builder, loc, 1));
  Value nextConsumer = arith::SelectOp::create(
      builder, loc, completionSuccess, advancedConsumer, consumer);
  emit::storeThrough64(
      builder, loc, consumerState, nextConsumer, detached);
  Value consumer32 = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), nextConsumer);
  consumer32 = arith::AndIOp::create(
      builder, loc, consumer32, i32Constant(builder, loc, 0x00ffffff));
  Value cqDbrecValue = byteSwap32(builder, loc, consumer32);
  Value completionDbrec = pointerFromAddress(
      builder, loc, completionDbrecAddress, builder.getI32Type());
  // The CQ DBREC is a consumer-index hint the NIC reads at its leisure;
  // relaxed is enough. The completed head is consumed only by GPU threads
  // (capacity waits, drain), so it publishes at GPU scope.
  releaseStore32(builder, loc, completionDbrec, cqDbrecValue,
                 /*firstLaneOnly=*/!detached,
                 emit::StoreOrder::RelaxedSys);
  Value completedValue = arith::SelectOp::create(
      builder, loc, completionSuccess, batchEnd, completionBase);
  releaseStore64(builder, loc, completedState, completedValue,
                 /*firstLaneOnly=*/!detached,
                 emit::StoreOrder::ReleaseGpu);

  Value finalOpcodeValid = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, finalOpcode,
      i64Constant(builder, loc, 15));
  Value finalAvailable = arith::AndIOp::create(
      builder, loc, finalOwnerReady, finalOpcodeValid);
  Value unavailable = arith::XOrIOp::create(
      builder, loc, finalAvailable,
      arith::ConstantOp::create(
          builder, loc, builder.getI1Type(),
          builder.getBoolAttr(true)));
  Value opcodeError = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, finalOpcode,
      i64Constant(builder, loc, 0));
  Value errorCode = arith::SelectOp::create(
      builder, loc, unavailable, i64Constant(builder, loc, 1),
      arith::SelectOp::create(
          builder, loc, opcodeError, i64Constant(builder, loc, 2),
          i64Constant(builder, loc, 3)));
  Value reportedError = arith::SelectOp::create(
      builder, loc, completionSuccess, i64Constant(builder, loc, 0),
      errorCode);
  Value errorState =
      completionStatePointer(builder, loc, endpoint, /*word=*/3);
  emit::storeThrough64(
      builder, loc, errorState, reportedError, detached);
  return success();
}

// Self-reclaiming completion (the `self-reclaim` lowering option). The state a
// reclaim loop advances, the CQ consumer index and the completed head, is
// carried in registers: one program issues on the endpoint, so nothing else
// changes it while the loop runs, and every thread of the program computes
// the same values from the same CQEs (only the first lane stores them).
struct ReclaimState {
  Value consumer;
  Value completed;
  Value failed;
};

// Consume the CQE at `state.consumer`: wait for the NIC to write it, within
// the endpoint's poll timeout, then advance the consumer and lift the CQE's
// 16-bit WQE counter onto the completed head. RC completes a send queue in
// order, so one CQE retires its WQE and every WQE posted before it. A CQE
// that does not arrive in time, or reports an error, sets `failed` and records
// the error in endpoint state as the poll does.
ReclaimState consumeOneCompletion(OpBuilder &builder, Location loc,
                                  Value endpoint, ReclaimState state) {
  Value completionQueue = endpointWord(builder, loc, endpoint, 3);
  Value qpnAndCqeSize = endpointWord(builder, loc, endpoint, 9);
  Value countAndVersion = endpointWord(builder, loc, endpoint, 10);
  Value cqeSize = high32(builder, loc, qpnAndCqeSize);
  Value cqeCount = low32(builder, loc, countAndVersion);
  Value count64 = zeroExtend64(builder, loc, cqeCount);
  Value size64 = zeroExtend64(builder, loc, cqeSize);
  Value entry = ringSlot(builder, loc, state.consumer, count64);
  Value entryOffset = arith::MulIOp::create(builder, loc, entry, size64);
  Value is128 = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, cqeSize,
      i32Constant(builder, loc, 128));
  Value cqe64Offset = arith::SelectOp::create(
      builder, loc, is128, i64Constant(builder, loc, 64),
      i64Constant(builder, loc, 0));
  Value cqeAddress = arith::AddIOp::create(
      builder, loc, completionQueue,
      arith::AddIOp::create(builder, loc, entryOffset, cqe64Offset));
  Value opOwnPointer = pointerFromAddress(
      builder, loc, addressOffset(builder, loc, cqeAddress, 63),
      builder.getI8Type());
  Value counterPointer = pointerFromAddress(
      builder, loc, addressOffset(builder, loc, cqeAddress, 60),
      builder.getI16Type());
  // The owner bit flips every lap of the ring (see lowerCompletion).
  Value expectedOwner = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(),
      arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ne,
          arith::AndIOp::create(builder, loc, state.consumer, count64),
          i64Constant(builder, loc, 0)));
  Value timeoutCycles = endpointWord(builder, loc, endpoint, 7);
  Value pollStart = readClock64(builder, loc);
  Value trueValue = arith::ConstantOp::create(
      builder, loc, builder.getI1Type(), builder.getBoolAttr(true));

  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value opOwn = loadValue(builder, loc, opOwnPointer, /*isVolatile=*/true);
  Value opOwn64 = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(), opOwn);
  Value owner = arith::AndIOp::create(
      builder, loc, opOwn64, i64Constant(builder, loc, 1));
  Value opcode = arith::ShRUIOp::create(
      builder, loc, opOwn64, i64Constant(builder, loc, 4));
  Value available = arith::AndIOp::create(
      builder, loc,
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, owner,
                            expectedOwner),
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, opcode,
                            i64Constant(builder, loc, 15)));
  Value elapsed = arith::SubIOp::create(
      builder, loc, readClock64(builder, loc), pollStart);
  Value timedOut = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, elapsed, timeoutCycles);
  Value keepWaiting = arith::XOrIOp::create(
      builder, loc, arith::OrIOp::create(builder, loc, available, timedOut),
      trueValue);
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);

  Value finalOpOwn64 = arith::ExtUIOp::create(
      builder, loc, builder.getI64Type(),
      loadValue(builder, loc, opOwnPointer, /*isVolatile=*/true));
  Value finalOpcode = arith::ShRUIOp::create(
      builder, loc, finalOpOwn64, i64Constant(builder, loc, 4));
  Value finalOwnerReady = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq,
      arith::AndIOp::create(builder, loc, finalOpOwn64,
                            i64Constant(builder, loc, 1)),
      expectedOwner);
  Value finalAvailable = arith::AndIOp::create(
      builder, loc, finalOwnerReady,
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                            finalOpcode, i64Constant(builder, loc, 15)));
  Value opcodeOk = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, finalOpcode,
      i64Constant(builder, loc, 0));
  Value success = arith::AndIOp::create(builder, loc, finalAvailable,
                                        opcodeOk);
  Value counter64 = zeroExtend64(
      builder, loc,
      byteSwap16To32(builder, loc,
                     loadValue(builder, loc, counterPointer,
                               /*isVolatile=*/true)));
  // The retired head is the WQE after the counter's, which is at most one
  // queue depth (< 2^16 WQEBBs, enforced at endpoint creation) ahead of the
  // head already retired.
  Value advance = arith::AndIOp::create(
      builder, loc,
      arith::SubIOp::create(
          builder, loc,
          arith::AddIOp::create(builder, loc, counter64,
                                i64Constant(builder, loc, 1)),
          state.completed),
      i64Constant(builder, loc, 0xffff));
  Value lifted = arith::AddIOp::create(builder, loc, state.completed, advance);
  Value nextConsumer = arith::SelectOp::create(
      builder, loc, success,
      arith::AddIOp::create(builder, loc, state.consumer,
                            i64Constant(builder, loc, 1)),
      state.consumer);
  Value nextCompleted =
      arith::SelectOp::create(builder, loc, success, lifted, state.completed);
  Value failed = arith::XOrIOp::create(builder, loc, success, trueValue);
  auto onFailure = scf::IfOp::create(builder, loc, TypeRange{}, failed,
                                     /*withElseRegion=*/false);
  {
    OpBuilder failure(onFailure.thenYield());
    Value errorCode = arith::SelectOp::create(
        failure, loc, finalAvailable, i64Constant(failure, loc, 2),
        i64Constant(failure, loc, 1));
    emit::storeThrough64(
        failure, loc, completionStatePointer(failure, loc, endpoint, 3),
        errorCode);
  }
  return {nextConsumer, nextCompleted, failed};
}

// Consume CQEs until `done(completed)` holds or one fails. The state is
// loaded once and carried through the loop; if anything was consumed, the
// consumer index, the CQ DBREC (so the NIC may reuse the entries) and the
// completed head are published once afterwards.
template <typename Done>
void reclaimUntil(OpBuilder &builder, Location loc, Value endpoint,
                  Done done) {
  Value consumerState = completionStatePointer(builder, loc, endpoint, 0);
  Value completedState = completionStatePointer(builder, loc, endpoint, 1);
  Value consumer = loadValue(builder, loc, consumerState, /*isVolatile=*/true);
  Value completed =
      loadValue(builder, loc, completedState, /*isVolatile=*/true);
  Value noFailure = arith::ConstantOp::create(
      builder, loc, builder.getI1Type(), builder.getBoolAttr(false));
  Value trueValue = arith::ConstantOp::create(
      builder, loc, builder.getI1Type(), builder.getBoolAttr(true));
  SmallVector<Type> types{builder.getI64Type(), builder.getI64Type(),
                          builder.getI1Type()};
  SmallVector<Location> locations(types.size(), loc);
  auto loop = scf::WhileOp::create(builder, loc, types,
                                   ValueRange{consumer, completed, noFailure});
  Block *before = builder.createBlock(&loop.getBefore(), {}, types, locations);
  builder.setInsertionPointToStart(before);
  Value finished = done(builder, before->getArgument(1));
  Value keepGoing = arith::AndIOp::create(
      builder, loc, arith::XOrIOp::create(builder, loc, finished, trueValue),
      arith::XOrIOp::create(builder, loc, before->getArgument(2), trueValue));
  scf::ConditionOp::create(builder, loc, keepGoing, before->getArguments());
  Block *after = builder.createBlock(&loop.getAfter(), {}, types, locations);
  builder.setInsertionPointToStart(after);
  ReclaimState next = consumeOneCompletion(
      builder, loc, endpoint,
      {after->getArgument(0), after->getArgument(1), after->getArgument(2)});
  scf::YieldOp::create(builder, loc,
                       ValueRange{next.consumer, next.completed, next.failed});
  builder.setInsertionPointAfter(loop);

  Value finalConsumer = loop.getResult(0);
  Value consumed = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, finalConsumer, consumer);
  auto publish = scf::IfOp::create(builder, loc, TypeRange{}, consumed,
                                   /*withElseRegion=*/false);
  OpBuilder published(publish.thenYield());
  emit::storeThrough64(published, loc, consumerState, finalConsumer);
  Value consumer32 = arith::AndIOp::create(
      published, loc,
      arith::TruncIOp::create(published, loc, published.getI32Type(),
                              finalConsumer),
      i32Constant(published, loc, 0x00ffffff));
  Value completionDbrec = pointerFromAddress(
      published, loc, endpointWord(published, loc, endpoint, 4),
      published.getI32Type());
  // As in lowerCompletion: the CQ DBREC is a hint the NIC reads at its
  // leisure, and only GPU threads read the completed head.
  releaseStore32(published, loc, completionDbrec,
                 byteSwap32(published, loc, consumer32),
                 /*firstLaneOnly=*/true, emit::StoreOrder::RelaxedSys);
  releaseStore64(published, loc, completedState, loop.getResult(1),
                 /*firstLaneOnly=*/true, emit::StoreOrder::ReleaseGpu);
}

// The capacity wait of a reclaiming module: while the queue cannot hold the
// batch, retire the oldest completed submissions from the CQ instead of
// waiting for a poll to do it.
void reclaimForCapacity(OpBuilder &builder, Location loc, Value endpoint,
                        Value batchEnd, Value sendWqeCount) {
  Value capacity = zeroExtend64(builder, loc, sendWqeCount);
  reclaimUntil(builder, loc, endpoint, [&](OpBuilder &b, Value completed) {
    Value availableEnd = arith::AddIOp::create(b, loc, completed, capacity);
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, batchEnd,
                                 availableEnd)
        .getResult();
  });
}

// A poll in a reclaiming module: consume CQEs, its own last, until the batch
// is retired. Earlier submissions' CQEs (requested by selfReclaim batches)
// may precede its own on the CQ.
LogicalResult lowerReclaimingCompletion(Operation *op, Location loc,
                                        Value endpoint, Value batchEnd,
                                        Value predicate) {
  if (failed(validateEndpoint(endpoint, op)))
    return failure();
  OpBuilder builder(op);
  if (predicate) {
    auto conditional = scf::IfOp::create(builder, loc, TypeRange{}, predicate,
                                         /*withElseRegion=*/false);
    builder.setInsertionPoint(conditional.thenBlock()->getTerminator());
  }
  reclaimUntil(builder, loc, endpoint, [&](OpBuilder &b, Value completed) {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, completed,
                                 batchEnd)
        .getResult();
  });
  return success();
}

LogicalResult lowerGroupWait(WaitGroupOp op, Value endpoint,
                             Value groupBase,
                             emit::StoreOrder doorbellOrder) {
  if (failed(validateEndpoint(endpoint, op)))
    return failure();
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value completionWqe = reserveBatch(builder, loc, endpoint, 1);
  Value groupEnd = arith::AddIOp::create(
      builder, loc, completionWqe, i64Constant(builder, loc, 1));
  Value layout = endpointWord(builder, loc, endpoint, 8);
  Value sendWqeCount = high32(builder, loc, layout);
  waitForQueueCapacity(
      builder, loc, endpoint, groupEnd, sendWqeCount);
  if (failed(writeCompletionWqe(op, loc, endpoint, completionWqe)))
    return failure();

  Value readyState =
      completionStatePointer(builder, loc, endpoint, /*word=*/2);
  waitUntilEqual(builder, loc, readyState, completionWqe);
  if (failed(lowerSubmit(op, loc, endpoint, groupEnd, /*detached=*/false,
                         doorbellOrder)))
    return failure();
  // Only other GPU threads read the ready head.
  releaseStore64(builder, loc, readyState, groupEnd, /*firstLaneOnly=*/true,
                 emit::StoreOrder::ReleaseGpu);
  return lowerCompletion(op, loc, endpoint, groupBase, groupEnd);
}

LogicalResult lowerDrain(DrainOp op, const DeviceLibrary *library,
                         bool detached, bool exclusive,
                         emit::StoreOrder doorbellOrder) {
  auto materializedEndpoint = materializeEndpoint(op.getEndpoint(), op);
  if (failed(materializedEndpoint))
    return failure();
  Value endpoint = *materializedEndpoint;
  if (failed(validateEndpoint(endpoint, op)))
    return failure();

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value issuer;
  if (op.getIssuerScope() == "thread") {
    issuer = arith::ConstantOp::create(
        builder, loc, builder.getI1Type(),
        builder.getBoolAttr(true));
  } else if (op.getIssuerScope() == "cta") {
    issuer = isFirstThread(builder, loc);
  } else {
    return op.emitError("unsupported drain issuer scope ")
           << op.getIssuerScope();
  }
  if (library) {
    Value address = pointerAddress(builder, loc, endpoint);
    library->call(
        builder, loc, issuer,
        ::aboleth::abi::kRdmaSymbolDrain,
        ValueRange{
            address,
            i64Constant(builder, loc, 0),
        });
    return success();
  }
  auto guard = scf::IfOp::create(
      builder, loc, issuer, /*withElseRegion=*/false);
  Operation *anchor = guard.thenYield();
  OpBuilder guarded(anchor);
  Value completedState =
      completionStatePointer(guarded, loc, endpoint, /*word=*/1);
  Value groupBase =
      loadValue(guarded, loc, completedState, /*isVolatile=*/true);
  // The leader guard cannot contain Triton's collective scalar-atomic
  // broadcast. Reuse the thread-local reservation emitter for this case.
  Value completionWqe = reserveBatch(
      guarded, loc, endpoint, 1, triton::MemSyncScope::GPU,
      detached || op.getIssuerScope() == "cta", exclusive);
  Value groupEnd = arith::AddIOp::create(
      guarded, loc, completionWqe, i64Constant(guarded, loc, 1));
  Value layout = endpointWord(guarded, loc, endpoint, 8);
  Value sendWqeCount = high32(guarded, loc, layout);
  waitForQueueCapacity(guarded, loc, endpoint, groupEnd, sendWqeCount);
  if (failed(writeCompletionWqe(anchor, loc, endpoint, completionWqe,
                                detached)))
    return failure();

  Value readyState =
      completionStatePointer(guarded, loc, endpoint, /*word=*/2);
  if (!exclusive)
    waitUntilEqual(guarded, loc, readyState, completionWqe);
  if (failed(lowerSubmit(anchor, loc, endpoint, groupEnd, detached,
                         doorbellOrder)))
    return failure();
  releaseStore64(guarded, loc, readyState, groupEnd,
                 /*firstLaneOnly=*/!detached,
                 emit::StoreOrder::ReleaseGpu);
  return lowerCompletion(anchor, loc, endpoint, groupBase, groupEnd,
                         /*predicate=*/{}, /*coalesced=*/false, detached);
}

LogicalResult lowerSystemFence(SystemFenceOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  auto guard = scf::IfOp::create(
      builder, loc, op.getPredicate(), /*withElseRegion=*/false);
  builder.setInsertionPoint(guard.thenYield());
  OperationState state(
      loc, triton::ElementwiseInlineAsmOp::getOperationName());
  state.addAttribute(
      "asm_string",
      builder.getStringAttr(
          "fence.acq_rel.sys;\nmov.u32 $0, 0;"));
  state.addAttribute(
      "constraints", builder.getStringAttr("=r"));
  state.addAttribute("pure", builder.getBoolAttr(false));
  state.addAttribute(
      "packed_element", builder.getI32IntegerAttr(1));
  state.addTypes(builder.getI32Type());
  builder.create(state);
  return success();
}

LogicalResult lowerPublishHead(PublishHeadOp op,
                               const DeviceLibrary *library, bool detached,
                               emit::StoreOrder doorbellOrder) {
  auto materializedEndpoint = materializeEndpoint(op.getEndpoint(), op);
  if (failed(materializedEndpoint))
    return failure();
  Value endpoint = *materializedEndpoint;
  if (failed(validateEndpoint(endpoint, op)))
    return failure();

  OpBuilder builder(op);
  Location loc = op.getLoc();
  // Only this GPU's issuers write the producer index and the ready head.
  Value head = acquireLoad64(
      builder, loc, producerIndexPointer(builder, loc, endpoint),
      /*gpuScope=*/true);
  Value readyState =
      completionStatePointer(builder, loc, endpoint, /*word=*/2);
  Value published =
      acquireLoad64(builder, loc, readyState, /*gpuScope=*/true);
  Value advanced = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, head, published);
  Value unconditional = arith::XOrIOp::create(
      builder, loc, op.getOnlyIfAdvanced(),
      arith::ConstantOp::create(
          builder, loc, builder.getI1Type(),
          builder.getBoolAttr(true)));
  Value shouldPublish =
      arith::OrIOp::create(builder, loc, unconditional, advanced);

  auto guard = scf::IfOp::create(
      builder, loc, shouldPublish, /*withElseRegion=*/false);
  Operation *anchor = guard.thenYield();
  OpBuilder guarded(anchor);
  if (library) {
    Value endpointAddress = pointerAddress(guarded, loc, endpoint);
    library->call(
        guarded, loc,
        arith::ConstantOp::create(
            guarded, loc, guarded.getI1Type(),
            guarded.getBoolAttr(true)),
        ::aboleth::abi::kRdmaSymbolPublishHead,
        ValueRange{
            endpointAddress,
            i64Constant(guarded, loc, 0),
            head,
        });
    return success();
  }
  if (failed(lowerSubmit(anchor, loc, endpoint, head, detached,
                         doorbellOrder)))
    return failure();
  releaseStore64(guarded, loc, readyState, head,
                 /*firstLaneOnly=*/!detached,
                 emit::StoreOrder::ReleaseGpu);
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
    return op.emitError("RDMA wait_signal requires an i64 signal");

  auto loop =
      scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
  Block *before = builder.createBlock(&loop.getBefore());
  builder.setInsertionPointToStart(before);
  Value observed = acquireLoad64(builder, loc, signal->pointer);
  Value keepWaiting = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, observed, op.getExpected());
  scf::ConditionOp::create(builder, loc, keepWaiting, ValueRange{});
  Block *after = builder.createBlock(&loop.getAfter());
  builder.setInsertionPointToStart(after);
  scf::YieldOp::create(builder, loc);
  return success();
}

} // namespace

void LowerRDMA::runOnOperation() {
  emission = {invariantDescriptors, vectorWqeStores};
  const DeviceLibrary library{bitcodePath};
  const DeviceLibrary *deviceLibrary =
      library.path.empty() ? nullptr : &library;
  if (detached && deviceLibrary) {
    getOperation().emitError(
        "detached fragments use the direct emitter; drop bitcode-path");
    return signalPassFailure();
  }
  const bool detachedFragment = detached;
  // The exclusive-issuer contract: every endpoint this module posts to has
  // exactly one issuing thread, so reservations drop their atomic and
  // publication drops the reservation-order spin. Opt-in; the caller owns
  // the single-producer proof (per-CTA endpoint fan-out, or a single-thread
  // Tier-1 caller).
  const bool exclusiveIssuer = exclusive;
  // The doorbell's release makes the WQEs, DBREC and payload visible to
  // whoever reads them after the ring. GPU scope puts them in L2, which is
  // where the NIC's reads of GPU memory are served, so it suffices when all of
  // them live in GPU memory; host-resident queues or payload need system
  // scope. World.compilation() sets it from the endpoints' queue placement.
  const emit::StoreOrder doorbellOrder =
      gpuScopeDoorbell ? emit::StoreOrder::ReleaseGpu
                       : emit::StoreOrder::ReleaseSys;
  if (exclusiveIssuer && deviceLibrary) {
    getOperation().emitError(
        "exclusive-issuer emission is a direct-emitter feature; "
        "drop bitcode-path");
    return signalPassFailure();
  }
  // Self-reclaim applies to a module with a selfReclaim poll (a wait whose
  // predicate is the kernel's own, so nothing bounds what it defers by the
  // queue) when the reclaim loops can own the CQ: one program issues on every
  // endpoint, every submission is reserved, published and polled by the
  // module itself, and every batch's completion is carried by a one-WQEBB
  // command. Any other module is lowered exactly as without the option.
  bool reclaimModule = false;
  if (selfReclaim && !deviceLibrary && !detachedFragment) {
    bool wanted = false;
    bool eligible = true;
    getOperation().walk([&](Operation *op) {
      if (auto poll = dyn_cast<PollOp>(op)) {
        wanted |= static_cast<bool>(poll.getSelfReclaim());
      } else if (auto submission = dyn_cast<SubmitOp>(op)) {
        auto last = submission.getTransfers().back().getDefiningOp<
            PostInlineOp>();
        if (!(exclusiveIssuer || submission.getExclusiveIssuer()) ||
            submission.getGroup() || submission.getDeferCompletion() ||
            submission.getHeadPublication() ||
            submission.getPlannedReservation() ||
            submission.getReservationScope() ||
            submission.getExternalPublication() ||
            submission.getDeferredAnnouncement() ||
            (last && inlineWqeBlocks(last.getPayloadBytes()) != 1))
          eligible = false;
      } else if (isa<BeginGroupOp, WaitGroupOp, DrainOp, PublishHeadOp>(op)) {
        eligible = false;
      }
    });
    reclaimModule = wanted && eligible;
    if (wanted && !eligible)
      getOperation().emitWarning(
          "self-reclaim: the module's endpoints are not single-issuer and "
          "module-owned; a predicated wait that defers more than the queue "
          "holds will wait for capacity indefinitely");
  }
  // Reserved send-queue range per submission, consumed by its poll.
  llvm::DenseMap<Operation *, std::pair<Value, Value>> submissionRanges;
  llvm::DenseMap<Operation *, std::pair<Value, Value>> groupStates;
  auto lowerSubmission = [&](SubmitOp submission,
                             PollOp poll) -> LogicalResult {
    OpBuilder submissionBuilder(submission);
    auto materializedEndpoint =
        materializeEndpoint(submission.getEndpoint(), submission);
    if (failed(materializedEndpoint))
      return failure();
    Value endpoint = *materializedEndpoint;
    Value endpointAddress =
        pointerAddress(submissionBuilder, submission.getLoc(), endpoint);
    // A planned reservation states that preflight covered the complete phase,
    // so producer advancement skips the capacity reload. The scope separately
    // records whether one CTA owns the queue.
    auto reservationScope = submission.getReservationScope();
    const bool ctaExclusive =
        reservationScope && *reservationScope == "cta";
    auto issuerScope = submission.getIssuerScope();
    const bool threadIssuer =
        issuerScope && *issuerScope == "thread";
    const bool plannedReservation =
        submission.getPlannedReservation() || ctaExclusive;
    const bool singleIssuer =
        exclusiveIssuer || static_cast<bool>(submission.getExclusiveIssuer());
    // A trailing no-op carries the batch's CQE, so no payload or signal
    // command requests one.
    const bool carriesCompletionNop = isa_and_nonnull<PostNopOp>(
        submission.getTransfers().back().getDefiningOp());

    // Send-queue slots are WQEBBs, not commands, and `batchSize` counts
    // commands because every other backend does. Each command's size here is a
    // compile-time constant, so the batch's WQEBB cost and each command's
    // offset inside the reservation are derived from the posts rather than
    // carried in Queue IR: a WQEBB is an mlx5 fact. Only an inline write can
    // exceed one WQEBB, so a batch without one reserves, indexes, publishes,
    // and polls exactly the numbers it did before.
    SmallVector<std::uint32_t> commandBlocks;
    std::uint32_t batchBlocks = 0;
    for (Value transfer : submission.getTransfers()) {
      commandBlocks.push_back(batchBlocks);
      auto inlinePut = transfer.getDefiningOp<PostInlineOp>();
      batchBlocks +=
          inlinePut ? inlineWqeBlocks(inlinePut.getPayloadBytes()) : 1;
    }

    Value lane;
    Value batchBase;
    if (deviceLibrary) {
      lane = threadIssuer
          ? arith::ConstantOp::create(
                submissionBuilder, submission.getLoc(),
                submissionBuilder.getI1Type(),
                submissionBuilder.getBoolAttr(true))
          : isFirstThread(submissionBuilder, submission.getLoc());
      Value count = i32Constant(
          submissionBuilder, submission.getLoc(), batchBlocks);
      if (plannedReservation) {
        batchBase = deviceLibrary->call(
            submissionBuilder, submission.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolReservePlanned,
            ValueRange{
                endpointAddress,
                count,
                i32Constant(
                    submissionBuilder, submission.getLoc(),
                    ctaExclusive ? 1 : 0),
            });
      } else {
        batchBase = deviceLibrary->call(
            submissionBuilder, submission.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolReserveBatch,
            ValueRange{endpointAddress, count});
      }
    } else {
      batchBase = reserveBatch(
          submissionBuilder, submission.getLoc(), endpoint, batchBlocks,
          ctaExclusive ? triton::MemSyncScope::CTA
                       : triton::MemSyncScope::GPU,
          detachedFragment, singleIssuer);
    }
    Value batchEnd = arith::AddIOp::create(
        submissionBuilder, submission.getLoc(), batchBase,
        i64Constant(submissionBuilder, submission.getLoc(), batchBlocks));
    if (!deviceLibrary && !plannedReservation) {
      Value layout = endpointWord(
          submissionBuilder, submission.getLoc(), endpoint, 8);
      Value sendWqeCount =
          high32(submissionBuilder, submission.getLoc(), layout);
      if (reclaimModule)
        reclaimForCapacity(submissionBuilder, submission.getLoc(), endpoint,
                           batchEnd, sendWqeCount);
      else
        waitForQueueCapacity(submissionBuilder, submission.getLoc(),
                             endpoint, batchEnd, sendWqeCount);
    }
    Value noCompletion = arith::ConstantOp::create(
        submissionBuilder, submission.getLoc(),
        submissionBuilder.getI1Type(),
        submissionBuilder.getBoolAttr(false));
    Value signalCompletion = noCompletion;
    if (!submission.getGroup() && !submission.getDeferCompletion()) {
      if (!poll)
        return submission.emitError(
            "ungrouped RDMA submission has no completion poll");
      if (!carriesCompletionNop) {
        signalCompletion = poll.getPredicate();
        // A selfReclaim batch whose poll is not due still carries a CQE, so
        // a later capacity wait or poll can retire it.
        if (reclaimModule && poll.getSelfReclaim())
          signalCompletion = Value{};
        if (!signalCompletion)
          signalCompletion = arith::ConstantOp::create(
              submissionBuilder, submission.getLoc(),
              submissionBuilder.getI1Type(),
              submissionBuilder.getBoolAttr(true));
      }
    }

    auto producerIndexAt = [&](std::size_t index) {
      return arith::AddIOp::create(
          submissionBuilder, submission.getLoc(), batchBase,
          i64Constant(submissionBuilder, submission.getLoc(),
                      commandBlocks[index]));
    };
    // Only the batch's last command requests the CQE that retires it, whatever
    // its kind: a polled batch may end in a payload as well as a signal.
    auto completionAt = [&](std::size_t index) {
      return index + 1 == submission.getTransfers().size() ? signalCompletion
                                                           : noCompletion;
    };
    auto onPut = [&](PostPutOp put, std::size_t index) -> LogicalResult {
      Value producerIndex = producerIndexAt(index);
      Value requestCompletion = completionAt(index);
      auto source = materializeLocalRegion(put.getSource(), put);
      auto destination =
          materializeRemoteRegion(put.getDestination(), put);
      if (failed(source) || failed(destination))
        return failure();
      // An mlx5 data segment with byte_count=0 does not describe an empty
      // message. Keep its ordered slot as a NOP, including completion/dependency
      // bookkeeping, rather than issuing a data-segment RDMA write. This also
      // handles a dynamic empty route without requiring a frontend predicate.
      Value predicate = put.getPredicate();
      APInt constantBytes;
      OpBuilder predicateBuilder(submission);
      if (matchPattern(put.getBytes(), m_ConstantInt(&constantBytes))) {
        if (constantBytes.isZero())
          predicate = arith::ConstantIntOp::create(
              predicateBuilder, put.getLoc(), 0, 1);
      } else {
        Value nonempty = arith::CmpIOp::create(
            predicateBuilder, put.getLoc(), arith::CmpIPredicate::ne,
            put.getBytes(), i32Constant(predicateBuilder, put.getLoc(), 0));
        predicate = arith::AndIOp::create(
            predicateBuilder, put.getLoc(), predicate, nonempty);
      }
      if (deviceLibrary) {
        OpBuilder builder(submission);
        Value predicate32 = arith::ExtUIOp::create(
            builder, put.getLoc(), builder.getI32Type(),
            predicate);
        Value requestCompletion32 =
            arith::ExtUIOp::create(
                builder, put.getLoc(),
                builder.getI32Type(),
                requestCompletion);
        deviceLibrary->call(
            builder, put.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolPostWrite,
            ValueRange{
                endpointAddress,
                producerIndex,
                pointerAddress(
                    builder, put.getLoc(), source->pointer),
                source->key,
                pointerAddress(
                    builder, put.getLoc(), destination->pointer),
                destination->key,
                put.getBytes(),
                predicate32,
                requestCompletion32,
                i32Constant(builder, put.getLoc(), 0),
            });
        return success();
      }
      return writeWqe(
          submission, put.getLoc(), endpoint, destination->pointer,
          source->pointer, put.getBytes(), source->key, destination->key,
          predicate, producerIndex,
          requestCompletion, /*controlWord=*/0, detachedFragment);
    };
    auto onInline = [&](PostInlineOp put,
                        std::size_t index) -> LogicalResult {
      Value producerIndex = producerIndexAt(index);
      Value requestCompletion = completionAt(index);
      auto destination =
          materializeRemoteRegion(put.getDestination(), put);
      if (failed(destination))
        return failure();
      const std::uint64_t payloadBytes = put.getPayloadBytes();
      if (deviceLibrary) {
        // The device library's inline primitive is typed on a 64-bit value and
        // always builds an unconditional write. Anything else is the direct
        // emitter's job rather than a silent reinterpretation of the command.
        if (payloadBytes != sizeof(std::uint64_t))
          return put.emitError(
                     "the mlx5 device library posts only an 8-byte inline "
                     "payload; this command carries ")
                 << payloadBytes
                 << " bytes and needs the direct emitter";
        if (!isAlwaysActive(put.getPredicate()))
          return put.emitError(
              "the mlx5 device library cannot predicate an inline payload; "
              "this command needs the direct emitter");
        OpBuilder builder(submission);
        SmallVector<Value> words = packInlinePayload(
            builder, put.getLoc(), put.getValues(), /*wordCount=*/2);
        Value low = zeroExtend64(builder, put.getLoc(), words[0]);
        Value high = arith::ShLIOp::create(
            builder, put.getLoc(),
            zeroExtend64(builder, put.getLoc(), words[1]),
            i64Constant(builder, put.getLoc(), 32));
        Value payload =
            arith::OrIOp::create(builder, put.getLoc(), low, high);
        deviceLibrary->call(
            builder, put.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolPostInlineU64,
            ValueRange{
                endpointAddress,
                producerIndex,
                pointerAddress(
                    builder, put.getLoc(), destination->pointer),
                destination->key,
                payload,
                arith::ExtUIOp::create(builder, put.getLoc(),
                                       builder.getI32Type(),
                                       requestCompletion),
                i32Constant(builder, put.getLoc(), 0),
            });
        return success();
      }
      return writeInlinePayloadWqe(
          submission, put.getLoc(), endpoint, destination->pointer,
          put.getValues(), destination->key, put.getPredicate(),
          producerIndex, requestCompletion, detachedFragment);
    };
    auto onSignal = [&](PostSignalOp signal,
                        std::size_t index) -> LogicalResult {
      Value producerIndex = producerIndexAt(index);
      // A signal closing an earlier segment of the batch is not its end.
      Value requestCompletion = completionAt(index);
      OpBuilder builder(submission);
      auto signalValue =
          materializeSignalValue(builder, signal.getSource(), signal);
      auto destination =
          materializeRemoteRegion(signal.getDestination(), signal);
      if (failed(signalValue) || failed(destination))
        return failure();
      // A strong publication is a responder compare-and-swap unless it is
      // fenced, in which case a write whose ctrl fence orders it after the
      // payload already on the queue carries it. Every other signal is a plain
      // write. Writes carry the epoch inline, so no local cell takes part.
      // A strong signal chained after another strong signal executes after it
      // at the responder, which already ordered the payload before both, so a
      // plain write publishes it.
      const bool atomic = signal.isChecked();
      const bool fenced =
          signal.getFencedSignal() && !signal.isOrderedByPrevious();
      const std::uint32_t controlWord = fenced ? kStrongOrderWord : 0;
      if ((signal.getPredicate() || signal.getExpected()) && deviceLibrary)
        return signal.emitError(
            "an elided signal needs the direct emitter");
      if (deviceLibrary) {
        Value requestCompletion32 = arith::ExtUIOp::create(
            builder, signal.getLoc(), builder.getI32Type(),
            requestCompletion);
        Value destinationAddress = pointerAddress(
            builder, signal.getLoc(), destination->pointer);
        SmallVector<Value> arguments{endpointAddress, producerIndex,
                                     destinationAddress, destination->key,
                                     *signalValue, requestCompletion32};
        if (!atomic)
          arguments.push_back(
              i32Constant(builder, signal.getLoc(), controlWord));
        deviceLibrary->call(
            builder, signal.getLoc(), lane,
            atomic ? ::aboleth::abi::kRdmaSymbolPostAtomicSignal
                   : ::aboleth::abi::kRdmaSymbolPostInlineU64,
            arguments);
        return success();
      }
      if (atomic)
        return writeAtomicSignalWqe(
            submission, signal.getLoc(), endpoint, destination->pointer,
            *signalValue, destination->key, producerIndex,
            requestCompletion, detachedFragment, signal.getExpected(),
            signal.getPredicate());
      return writeInlineSignalWqe(
          submission, signal.getLoc(), endpoint, destination->pointer,
          *signalValue, destination->key, producerIndex, requestCompletion,
          controlWord, detachedFragment, signal.getPredicate());
    };
    auto onNop = [&](PostNopOp nop, std::size_t index) -> LogicalResult {
      Value producerIndex = producerIndexAt(index);
      if (deviceLibrary) {
        OpBuilder builder(submission);
        // The library's no-op always requests a CQE, as the direct emission's
        // completion WQE does.
        deviceLibrary->call(
            builder, nop.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolPostNop,
            ValueRange{endpointAddress, producerIndex});
        return success();
      }
      return writeCompletionWqe(
          submission, nop.getLoc(), endpoint, producerIndex,
          detachedFragment);
    };
    // TODO: warp-cooperative posting. Every WQE store above is predicated
    // to lane 0. A batch's shape is static, so its commands could be laid
    // out as a tensor and built one WQE per lane.
    if (failed(forEachTransfer(
            submission, onPut, onSignal, onNop, onInline)))
      return failure();

    // Batches ring the doorbell in reservation order so the queue is published
    // as a contiguous run no matter which producer filled it first. Head
    // publication drops that wait: the doorbell covers whatever the producer
    // index reached, which the caller's barrier proves is fully built.
    const bool publishHead =
        static_cast<bool>(submission.getHeadPublication());
    const bool externalPublication =
        static_cast<bool>(submission.getExternalPublication());
    if (externalPublication) {
      // The enclosing phase adapter publishes and completes this WQE.
    } else if (deviceLibrary) {
      if (publishHead) {
        Value head = acquireLoad64(
            submissionBuilder, submission.getLoc(),
            producerIndexPointer(
                submissionBuilder, submission.getLoc(), endpoint),
            /*gpuScope=*/true);
        deviceLibrary->call(
            submissionBuilder, submission.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolPublishHead,
            ValueRange{
                endpointAddress,
                i64Constant(
                    submissionBuilder, submission.getLoc(), 0),
                head,
            });
      } else {
        deviceLibrary->call(
            submissionBuilder, submission.getLoc(), lane,
            ::aboleth::abi::kRdmaSymbolPublish,
            ValueRange{
                endpointAddress,
                i64Constant(
                    submissionBuilder, submission.getLoc(), 0),
                batchBase,
                batchEnd,
                i32Constant(
                    submissionBuilder, submission.getLoc(), 0),
            });
      }
    } else {
      OpBuilder publishBuilder(submission);
      Value readyState = completionStatePointer(
          publishBuilder, submission.getLoc(), endpoint,
          /*word=*/2);
      if (publishHead) {
        Value head = acquireLoad64(
            publishBuilder, submission.getLoc(),
            producerIndexPointer(
                publishBuilder, submission.getLoc(), endpoint),
            /*gpuScope=*/true);
        if (failed(lowerSubmit(submission, submission.getLoc(), endpoint,
                               head, detachedFragment, doorbellOrder)))
          return failure();
        OpBuilder readyBuilder(submission);
        releaseStore64(
            readyBuilder, submission.getLoc(), readyState, head,
            /*firstLaneOnly=*/!detachedFragment,
            emit::StoreOrder::ReleaseGpu);
      } else {
        // An exclusive endpoint has exactly one issuing thread, so batches
        // are published in program order by construction — the
        // reservation-order spin serializes nobody and is dropped.
        if (!singleIssuer)
          waitUntilEqual(
              publishBuilder, submission.getLoc(), readyState, batchBase);
        // A deferred announcement rings only when the poll is due; the ready
        // head below still advances, so the next ring on the endpoint, which
        // covers everything up to its own end, announces these commands.
        // Only GPU threads consume the ready head during the kernel. Other
        // producers wait on it, so for them it is released after the ring,
        // keeping their rings (and the DBREC) monotonic. An exclusive issuer
        // is its only reader, so a relaxed store ahead of the ring, which the
        // doorbell's release then covers, spares the second fence, and a
        // deferred round that does not ring pays none.
        if (singleIssuer)
          releaseStore64(
              publishBuilder, submission.getLoc(), readyState, batchEnd,
              /*firstLaneOnly=*/!detachedFragment,
              emit::StoreOrder::RelaxedGpu);
        Operation *ringAt = submission;
        if (submission.getDeferredAnnouncement())
          ringAt = scf::IfOp::create(publishBuilder, submission.getLoc(),
                                     poll.getPredicate(),
                                     /*withElseRegion=*/false)
                       .thenYield();
        if (failed(lowerSubmit(ringAt, submission.getLoc(), endpoint,
                               batchEnd, detachedFragment, doorbellOrder)))
          return failure();
        if (!singleIssuer) {
          OpBuilder readyBuilder(submission);
          releaseStore64(
              readyBuilder, submission.getLoc(), readyState, batchEnd,
              /*firstLaneOnly=*/!detachedFragment,
              emit::StoreOrder::ReleaseGpu);
        }
      }
    }
    submissionRanges[submission] = {batchBase, batchEnd};
    return success();
  };

  auto lowerPollOp = [&](PollOp poll) -> LogicalResult {
    auto submission = poll.getSubmission().getDefiningOp<SubmitOp>();
    auto range = submissionRanges.find(submission);
    if (range == submissionRanges.end())
      return poll.emitError("submission has no reserved queue range");
    auto endpoint = materializeEndpoint(submission.getEndpoint(), poll);
    if (failed(endpoint))
      return failure();
    if (deviceLibrary) {
      OpBuilder builder(poll);
      Value address =
          pointerAddress(builder, poll.getLoc(), *endpoint);
      Value lane = isFirstThread(builder, poll.getLoc());
      if (Value predicate = poll.getPredicate())
        lane = arith::AndIOp::create(
            builder, poll.getLoc(), lane, predicate);
      deviceLibrary->call(
          builder, poll.getLoc(), lane,
          ::aboleth::abi::kRdmaSymbolPoll,
          ValueRange{
              address,
              range->second.first,
              range->second.second,
          });
      return success();
    }
    if (reclaimModule)
      return lowerReclaimingCompletion(poll, poll.getLoc(), *endpoint,
                                       range->second.second,
                                       poll.getPredicate());
    return lowerCompletion(poll, poll.getLoc(), *endpoint,
                           range->second.first, range->second.second,
                           poll.getPredicate(),
                           static_cast<bool>(poll.getCoalesced()),
                           detachedFragment);
  };

  auto lowerBeginGroupOp = [&](BeginGroupOp group) -> LogicalResult {
    // A group's begin-side scalar atomic has a live result Triton would
    // broadcast through shared memory, which a detached fragment may not
    // impose on its caller; nothing detached needs groups yet.
    if (detachedFragment)
      return group.emitError(
          "completion groups are not yet available in detached fragments");
    auto endpoint = materializeEndpoint(group.getEndpoint(), group);
    if (failed(endpoint))
      return failure();
    OpBuilder builder(group);
    Value base = reserveBatch(builder, group.getLoc(), *endpoint, 0);
    groupStates[group] = {*endpoint, base};
    return success();
  };

  auto lowerWaitGroupOp = [&](WaitGroupOp wait) -> LogicalResult {
    auto begin = wait.getGroup().getDefiningOp<BeginGroupOp>();
    if (!begin)
      return wait.emitError("completion group has no matching begin_group");
    auto state = groupStates.find(begin);
    if (state == groupStates.end())
      return wait.emitError("completion group has no direct RDMA state");
    return lowerGroupWait(wait, state->second.first, state->second.second,
                          doorbellOrder);
  };

  auto lowerDrainOp = [&](DrainOp drain) {
    return lowerDrain(drain, deviceLibrary, detachedFragment,
                      exclusiveIssuer ||
                          static_cast<bool>(drain.getExclusiveIssuer()),
                      doorbellOrder);
  };

  auto lowerSystemFenceOp = [&](SystemFenceOp fence) {
    return lowerSystemFence(fence);
  };

  auto lowerPublishHeadOp = [&](PublishHeadOp publication) {
    return lowerPublishHead(publication, deviceLibrary, detachedFragment,
                            doorbellOrder);
  };

  QueueLoweringHooks hooks;
  hooks.backendName = "RDMA";
  hooks.lowerSubmission = lowerSubmission;
  hooks.lowerPoll = lowerPollOp;
  hooks.lowerBeginGroup = lowerBeginGroupOp;
  hooks.lowerWaitGroup = lowerWaitGroupOp;
  hooks.lowerDrain = lowerDrainOp;
  hooks.lowerSystemFence = lowerSystemFenceOp;
  hooks.lowerPublishHead = lowerPublishHeadOp;
  hooks.lowerWaitSignal = lowerWaitSignal;
  if (failed(runQueueLowering(getOperation(), hooks)))
    signalPassFailure();
}

} // namespace mlir::aboleth
