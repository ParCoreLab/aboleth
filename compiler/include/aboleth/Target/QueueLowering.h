#pragma once

// The driver every backend lowering shares.
//
// Queue IR is the boundary between target-independent transport planning and
// target-specific lowering. Walking that IR, pairing submissions with their
// completion polls, retiring operations in dependency order, and proving no
// Aboleth operation survives are all properties of the boundary rather than of
// any one transport. Backends supply only the commands they build.

#include "aboleth/IR/Dialect.h"
#include <cstdint>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::aboleth {

struct QueueLoweringHooks {
  // Appears verbatim in diagnostics, e.g. "RDMA", "proxy", "P2P", "EFA".
  StringRef backendName;

  // Backends that erase tokens outright require every token to be dead by the
  // time the operation is retired.
  bool requireUnusedTokens = false;

  // Existing backend handle encodings remain byte-for-byte unchanged.
  std::uint64_t endpointKindMask = 3;

  // A completed-prefix signal is unchecked only because the lowering retires
  // its payload before publishing it. Other backends must reject that IR.
  bool supportsCompletedPrefix = false;

  // Lower one submission. The poll that completes it is supplied when the
  // submission is ungrouped and not deferred, and is null otherwise, so a
  // backend can request a completion entry only when one will be consumed.
  llvm::function_ref<LogicalResult(SubmitOp, PollOp)> lowerSubmission;

  // Remaining hooks may be left unset by backends that have nothing to emit
  // for that operation; the operation is still retired.
  llvm::function_ref<LogicalResult(PollOp)> lowerPoll;
  llvm::function_ref<LogicalResult(BeginGroupOp)> lowerBeginGroup;
  llvm::function_ref<LogicalResult(WaitGroupOp)> lowerWaitGroup;
  llvm::function_ref<LogicalResult(DrainOp)> lowerDrain;
  llvm::function_ref<LogicalResult(SystemFenceOp)> lowerSystemFence;
  llvm::function_ref<LogicalResult(PublishHeadOp)> lowerPublishHead;
  llvm::function_ref<LogicalResult(WaitSignalOp)> lowerWaitSignal;
};

// Visit a submission's transfers in verified operand order. Backends emit
// commands in this order rather than by independently walking operation kinds.
// A backend that leaves `onNop` or `onInline` unset rejects submissions
// carrying that command rather than silently dropping it.
LogicalResult forEachTransfer(
    SubmitOp submission,
    llvm::function_ref<LogicalResult(PostPutOp, std::size_t)> onPut,
    llvm::function_ref<LogicalResult(PostSignalOp, std::size_t)> onSignal,
    llvm::function_ref<LogicalResult(PostNopOp, std::size_t)> onNop = {},
    llvm::function_ref<LogicalResult(PostInlineOp, std::size_t)> onInline = {});

// Run one backend over every queue operation in the module and retire them.
LogicalResult runQueueLowering(ModuleOp module,
                               const QueueLoweringHooks &hooks);

} // namespace mlir::aboleth
