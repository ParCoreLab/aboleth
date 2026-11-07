#ifndef ABOLETH_TARGET_PASSES
#define ABOLETH_TARGET_PASSES

include "mlir/Pass/PassBase.td"

def AbolethLowerP2PPass
    : Pass<"aboleth-lower-p2p", "mlir::ModuleOp"> {
  let summary = "Lower scheduled Aboleth communication to direct P2P memory operations";
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"hipDeviceLibrary", "hip-device-library", "std::string",
           /*default=*/"\"\"", "HIP memory-ordering device bitcode; empty selects CUDA.">,
    Option<"detached", "detached", "bool",
           /*default=*/"false",
           "The module is a detached device fragment with no CTA context: "
           "copy-loop stores and signal atomics are emitted per calling "
           "thread instead of once per block, and nothing may impose a "
           "block barrier on the caller.">,
    Option<"copyMode", "copy-mode", "std::string",
           /*default=*/"\"\"",
           "Copy emission for put payloads: 'scalar' (byte loop), 'wide' "
           "(alignment-checked 8-byte loop), 'warp' (one-warp cooperative "
           "tensor copy with a warp-scope fence; the caller must enter "
           "with a full warp). Empty keeps the in-kernel behaviour, which "
           "reads the ABOLETH_P2P_* environment.">
  ];
}

def AbolethLowerRDMAPass
    : Pass<"aboleth-lower-rdma", "mlir::ModuleOp"> {
  let summary = "Lower scheduled Aboleth communication to mlx5 RDMA operations";
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"bitcodePath", "bitcode-path", "std::string",
           /*default=*/"\"\"",
           "Path to shared mlx5 device bitcode. Empty keeps the "
           "existing direct IR emitter.">,
    Option<"detached", "detached", "bool",
           /*default=*/"false",
           "The module is a detached device fragment with no CTA context: "
           "each calling thread is its own issuer, so the reservation is a "
           "per-thread inline atomic rather than a scalar tt.atomic_rmw "
           "(whose live result Triton broadcasts through shared memory), "
           "publishing stores drop their implicit tid.x==0 predicate, and "
           "completion groups are refused.">,
    Option<"exclusive", "exclusive", "bool",
           /*default=*/"false",
           "Every endpoint this module posts to has exactly one issuing "
           "thread (the caller's proof: per-CTA endpoint fan-out or a "
           "single-thread Tier-1 caller). Reservations become plain "
           "load/add/store and publication drops the reservation-order "
           "spin; the queue-capacity wait is kept.">,
    Option<"gpuScopeDoorbell", "gpu-scope-doorbell", "bool",
           /*default=*/"false",
           "Publish the doorbell with a GPU-scope release instead of a "
           "system-scope one. Valid only when every send queue, doorbell "
           "record and payload source lives in GPU memory.">,
    Option<"selfReclaim", "self-reclaim", "bool",
           /*default=*/"false",
           "In a module with a selfReclaim poll whose submissions all have "
           "one issuing program, every capacity wait consumes completions "
           "itself when the queue is full, selfReclaim batches always "
           "request their CQE, and every poll consumes CQEs until its batch "
           "is retired. Other modules are lowered as without it.">,
    Option<"invariantDescriptors", "invariant-descriptors", "bool",
           /*default=*/"false",
           "Read endpoint-descriptor words with pure non-coherent loads, "
           "which CSE shares across a batch. The host writes the descriptor "
           "once, at endpoint creation.">,
    Option<"vectorWqeStores", "vector-wqe-stores", "bool",
           /*default=*/"false",
           "Write each RDMA-write WQE as three 16-byte stores instead of "
           "ten scalar ones.">
  ];
}

def AbolethLowerProxyPass
    : Pass<"aboleth-lower-proxy", "mlir::ModuleOp"> {
  let summary = "Lower scheduled Aboleth communication to a host verbs proxy";
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"hipDeviceLibrary", "hip-device-library", "std::string",
           /*default=*/"\"\"", "HIP memory-ordering device bitcode; empty selects CUDA.">,
    Option<"detached", "detached", "bool", "false",
           "Emit a HIP per-caller device function without a shared-memory ABI">,
  ];
}

def AbolethLowerEFAPass
    : Pass<"aboleth-lower-efa", "mlir::ModuleOp"> {
  let summary = "Lower scheduled Aboleth communication to AWS EFA";
  let description = [{
    Emits calls to the efa-dp-direct device primitives, which are linked into
    the kernel as extern bitcode. SRD delivers out of order and offers no
    remote atomics, so a batch retires by counting completions and strong
    publication uses one of the two unordered legalizations.
  }];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"publicationScheme", "publication-scheme", "std::string",
           /*default=*/"\"\"",
           "Strong publication legalization: 'fence' retires payload "
           "completions before signalling, 'imm' counts receiver arrivals. "
           "Empty defers to ABOLETH_EFA_PUBLICATION.">,
    Option<"bitcodePath", "bitcode-path", "std::string",
           /*default=*/"\"\"",
           "Path to the EFA device bitcode. Empty defers to "
           "ABOLETH_EFA_BITCODE.">,
  ];
}

def AbolethLowerIonicPass : Pass<"aboleth-lower-ionic", "mlir::ModuleOp"> {
  let summary = "Lower Aboleth Queue IR to GPU-initiated Ionic RC queues";
  let dependentDialects = ["mlir::triton::TritonDialect",
                          "mlir::arith::ArithDialect", "mlir::scf::SCFDialect"];
  let options = [Option<"bitcodePath", "bitcode-path", "std::string", "\"\"",
                        "HIP Ionic device primitive bitcode">,
                 Option<"detached", "detached", "bool", "false",
                        "Each calling thread owns its transport effects">,
                 Option<"shareProducerFence", "share-producer-fence", "bool", "false",
                        "Share the reserve producer fence within a proved CTA submission">,
                 Option<"fencedPublication", "fenced-publication", "bool", "false",
                        "Realize planner-proved strict publication with a fenced Ionic write">];
}

#endif // ABOLETH_TARGET_PASSES
