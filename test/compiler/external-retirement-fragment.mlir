module {
  aboleth.transport_program {
    "aboleth.transport_fragment"() <{programId = 1 : i32,
        name = "retire", argumentNames = ["endpoint"]}> ({
    ^bb0(%argument: i64):
      %endpoint = aboleth.bind_endpoint %argument : i64 -> !aboleth.endpoint
      aboleth.drain %endpoint : !aboleth.endpoint
      ub.unreachable
    }) : () -> ()
    ub.unreachable
  } {name = "retirement"}
}
