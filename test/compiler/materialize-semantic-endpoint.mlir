module {
  tt.func public @retire(%table: !tt.ptr<i64>) {
    %pid = tt.get_program_id x : i32
    %endpoint = aboleth.endpoint %table[%pid] : !tt.ptr<i64> -> !aboleth.endpoint
    %id = arith.constant 1 : i32
    "aboleth.invoke_transport_phase"(%id, %endpoint) : (i32, !aboleth.endpoint) -> ()
    tt.return
  }
}
// CHECK-LABEL: tt.func public @retire
// CHECK: %[[ENDPOINT:.*]] = aboleth.endpoint
// CHECK-NOT: aboleth.bind_endpoint
// CHECK: aboleth.drain %[[ENDPOINT]]
// CHECK-NOT: aboleth.invoke_transport_phase
