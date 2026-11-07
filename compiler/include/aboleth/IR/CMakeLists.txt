set(LLVM_TARGET_DEFINITIONS AbolethDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=aboleth)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=aboleth)

set(LLVM_TARGET_DEFINITIONS AbolethTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=aboleth)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=aboleth)

set(LLVM_TARGET_DEFINITIONS AbolethOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_public_tablegen_target(AbolethTableGen)
