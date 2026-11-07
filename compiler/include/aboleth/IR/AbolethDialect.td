#ifndef ABOLETH_DIALECT
#define ABOLETH_DIALECT

include "mlir/IR/OpBase.td"

def Aboleth_Dialect : Dialect {
  let name = "aboleth";
  let cppNamespace = "::mlir::aboleth";
  let description = [{
    Communication operations preserved for Aboleth compiler optimization.
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif // ABOLETH_DIALECT
