// Detached P2P fragments have no CTA context: every store lands from the
// calling thread (Triton predicates scalar tt.store on lane 0 of the block,
// which is wrong when each caller owns its own copy), the signal is a
// per-thread release exchange rather than a once-per-block tt.atomic_rmw,
// no lowering may impose a block barrier on the caller, and `quiet` is
// vacuous because a p2p put completes when its stores retire.
//
// RUN: aboleth-opt %s --pass form-batches --pass lower-p2p=,detached \
// RUN:   | FileCheck %s --check-prefix=WIDE
// RUN: aboleth-opt %s --pass form-batches --pass lower-p2p=scalar,detached \
// RUN:   | FileCheck %s --check-prefix=SCALAR
// RUN: aboleth-opt %s --pass form-batches --pass lower-p2p=warp,detached \
// RUN:   | FileCheck %s --check-prefix=WARP

module {
  tt.func public @producer(
      %endpoint: !tt.ptr<i8>,
      %destination: !tt.ptr<i8>,
      %source: !tt.ptr<i8>,
      %remote_signal: !tt.ptr<i64>,
      %local_signal: !tt.ptr<i64>,
      %bytes: i32,
      %source_key: i32,
      %destination_key: i32,
      %predicate: i1) {
    %ep = aboleth.bind_endpoint %endpoint
      : !tt.ptr<i8> -> !aboleth.endpoint
    %dst = aboleth.bind_remote_region %destination, %destination_key
      : (!tt.ptr<i8>, i32) -> !aboleth.remote_region
    %src = aboleth.bind_local_region %source, %source_key
      : (!tt.ptr<i8>, i32) -> !aboleth.local_region
    %signal_dst = aboleth.bind_remote_region %remote_signal, %destination_key
      : (!tt.ptr<i64>, i32) -> !aboleth.remote_region
    %signal_src = aboleth.bind_local_region %local_signal, %source_key
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    %0 = aboleth.put %ep, %dst, %src, %bytes, %predicate
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, i32, i1) -> !aboleth.token
    %1 = aboleth.signal %ep, %signal_dst, %signal_src, %0
      : (!aboleth.endpoint, !aboleth.remote_region, !aboleth.local_region, !aboleth.token) -> !aboleth.token
    aboleth.wait %1 : !aboleth.token
    tt.return
  }

  tt.func public @drain(%endpoint_ptr: !tt.ptr<i8>) {
    %endpoint = aboleth.bind_endpoint %endpoint_ptr
      : !tt.ptr<i8> -> !aboleth.endpoint
    aboleth.drain %endpoint {issuerScope = "thread"} : !aboleth.endpoint
    tt.return
  }

  tt.func public @consumer(
      %signal: !tt.ptr<i64>,
      %expected: i64) {
    %zero = arith.constant 0 : i32
    %local = aboleth.bind_local_region %signal, %zero
      : (!tt.ptr<i64>, i32) -> !aboleth.local_region
    aboleth.wait_signal %local, %expected
      : (!aboleth.local_region, i64) -> ()
    tt.return
  }
}

// Empty copy mode falls back to 'wide' when detached: the alignment check
// stays, every loop body stores through per-thread inline asm, and nothing
// reads the build machine's ABOLETH_P2P_* environment (no cooperative
// tensor path, no block barrier). 16-byte-aligned copies move 64 bytes per
// iteration (four 16-byte loads ahead of their stores), the 8-byte loop
// taking the tail and the 8-byte-only-aligned case.
// WIDE-LABEL: tt.func public @producer
// WIDE: arith.ori
// WIDE: arith.andi
// WIDE: arith.cmpi eq
// WIDE: scf.if
// WIDE-NOT: tt.make_range
// WIDE-NOT: "bar.sync 0;
// WIDE: arith.constant 15 : i64
// WIDE: arith.constant -64 : i32
// WIDE: tt.elementwise_inline_asm "{ .reg .b64 w<8>; ld.global.v2.u64 {w0, w1}, [$1];
// WIDE-SAME: st.global.L1::no_allocate.v2.u64 [$2+48], {w6, w7};
// WIDE: tt.elementwise_inline_asm "{ mov.b64 $0, $2; st.global.L1::no_allocate.b64 [$1], $2; }"
// WIDE: tt.elementwise_inline_asm "{ mov.b64 $0, $2; st.global.L1::no_allocate.b64 [$1], $2; }"
// WIDE: tt.elementwise_inline_asm "{ mov.b32 $0, $2; st.global.b8 [$1], $2; }"
// WIDE-NOT: tt.atomic_rmw
// WIDE: tt.elementwise_inline_asm "atom.release.sys.global.exch.b64
// WIDE-NOT: "bar.sync 0;
// WIDE-LABEL: tt.func public @drain
// WIDE-NOT: aboleth.
// WIDE-LABEL: tt.func public @consumer
// WIDE: ld.acquire.sys.global
// WIDE-NOT: aboleth.

// Scalar mode is a straight byte loop under the predicate conditional: no
// alignment branch at all.
// SCALAR-LABEL: tt.func public @producer
// SCALAR: scf.if
// SCALAR-NOT: arith.ori
// SCALAR-NOT: arith.cmpi eq
// SCALAR: scf.for
// SCALAR: tt.elementwise_inline_asm "{ mov.b32 $0, $2; st.global.b8 [$1], $2; }"
// SCALAR-NOT: st.global.L1::no_allocate.b64
// SCALAR-NOT: tt.atomic_rmw
// SCALAR: tt.elementwise_inline_asm "atom.release.sys.global.exch.b64
// SCALAR-LABEL: tt.func public @drain
// SCALAR-NOT: aboleth.
// SCALAR-LABEL: tt.func public @consumer
// SCALAR-NOT: aboleth.

// Warp mode keeps the cooperative tensor copy (the mask covers the partial
// chunk, so no scalar tail store exists to be lane-predicated) but fences at
// warp scope: the caller's contract is to enter with a full warp, and a
// detached function may never impose a block barrier.
// WARP-LABEL: tt.func public @producer
// WARP: scf.if
// WARP: tt.make_range
// WARP-SAME: tensor<2048xi32>
// WARP: tt.load {{.*}} : tensor<2048x!tt.ptr<i64>>
// WARP: tt.store {{.*}} : tensor<2048x!tt.ptr<i64>>
// WARP-NOT: "bar.sync 0;
// WARP: tt.elementwise_inline_asm "bar.warp.sync 0xffffffff;
// WARP: tt.elementwise_inline_asm "{ mov.b32 $0, $2; st.global.b8 [$1], $2; }"
// WARP-NOT: tt.atomic_rmw
// WARP: tt.elementwise_inline_asm "atom.release.sys.global.exch.b64
// WARP-NOT: "bar.sync 0;
// WARP-LABEL: tt.func public @drain
// WARP-NOT: aboleth.
// WARP-LABEL: tt.func public @consumer
// WARP-NOT: aboleth.
