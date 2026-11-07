set(LLVM_TARGET_DEFINITIONS TargetPasses.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AbolethTarget)
add_public_tablegen_target(AbolethTargetPassesIncGen)
