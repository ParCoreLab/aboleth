#ifndef ABOLETH_TYPES
#define ABOLETH_TYPES

include "mlir/IR/AttrTypeBase.td"
include "aboleth/IR/AbolethDialect.td"

class Aboleth_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<Aboleth_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def Aboleth_TokenType : Aboleth_TypeDef<"Token", "token"> {
  let summary = "Ordered Aboleth communication transfer";
}

def Aboleth_EndpointType : Aboleth_TypeDef<"Endpoint", "endpoint"> {
  let summary = "Backend-neutral ordered communication endpoint";
}

def Aboleth_LogicalRegionType
    : Aboleth_TypeDef<"LogicalRegion", "logical_region"> {
  let summary = "Frontend-neutral logical communication region or view";
  let description = [{
    An SSA handle for communication dataflow. Shape, layout, visibility,
    mutability, and escape facts live on the defining region or view
    operation. This type is intentionally separate from executable local and
    remote registered-memory region handles.
  }];
}

def Aboleth_WorksetType
    : Aboleth_TypeDef<"Workset", "workset"> {
  let summary = "Runtime collection of typed communication work items";
  let description = [{
    Describes bounded runtime communication work through immutable field
    regions such as source row, destination peer, destination row, byte count,
    and predicate. The workset is logical compiler input, not an executable
    descriptor layout.
  }];
}

def Aboleth_LocalRegionType
    : Aboleth_TypeDef<"LocalRegion", "local_region"> {
  let summary = "Backend-neutral local registered memory region";
}

def Aboleth_RemoteRegionType
    : Aboleth_TypeDef<"RemoteRegion", "remote_region"> {
  let summary = "Backend-neutral remote registered memory region";
}

def Aboleth_GroupType : Aboleth_TypeDef<"Group", "group"> {
  let summary = "Backend-neutral completion group for one endpoint";
}

#endif // ABOLETH_TYPES
