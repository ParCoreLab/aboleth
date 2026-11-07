// A fragment is cloned into the kernel that invokes it, out of a plan module
// that the compilation parses, reads, and destroys. A body that names a value
// defined outside itself would be cloned with an operand pointing into that
// dead module, so every input has to arrive as a block argument.
//
// RUN: not aboleth-opt %s 2>&1 | FileCheck %s

module {
  "aboleth.transport_program"() ({
  ^bb0(%outer: i32):
    "aboleth.transport_fragment"() <{
      programId = 7 : i32,
      name = "dispatch",
      argumentNames = ["endpoint", "destination", "destination_key",
                       "predicate"]
    }> ({
    ^bb0(
        %endpoint: i64,
        %destination: i64,
        %destinationKey: i32,
        %predicate: i1):
      %ep = aboleth.bind_endpoint %endpoint
        : i64 -> !aboleth.endpoint
      %dst = aboleth.bind_remote_region %destination, %destinationKey
        : (i64, i32) -> !aboleth.remote_region
      // %outer belongs to the plan module, not to this fragment.
      %put = aboleth.put_inline
        %ep, %dst, [%outer], %predicate
        : (!aboleth.endpoint, !aboleth.remote_region, i32, i1)
          -> !aboleth.token
      ub.unreachable
    }) : () -> ()
    ub.unreachable
  }) {
    name = "capturing-fragment"
  } : () -> ()
}

// CHECK: error: 'aboleth.put_inline' op uses a value defined outside its transport fragment
