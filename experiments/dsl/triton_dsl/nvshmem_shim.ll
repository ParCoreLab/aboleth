; NVSHMEM's device API at Triton's extern ABI: scalar i64/i32 arguments and an
; i64 result, so upstream Triton's extern_elementwise can call it (pointers
; and void returns are not expressible there; Triton-distributed adds its own
; extern call for them). Each wrapper only converts and calls, and is
; always-inlined, so the kernel holds exactly the NVSHMEM calls the CUDA arm
; makes (comm_nvshmem.cuh). Linked before libnvshmem_device_sm_*.bc.
target triple = "nvptx64-nvidia-cuda"

declare void @nvshmem_putmem_nbi(ptr, ptr, i64, i32)
declare void @nvshmemx_putmem_nbi_block(ptr, ptr, i64, i32)
declare void @nvshmem_putmem_signal_nbi(ptr, ptr, i64, ptr, i64, i32, i32)
declare void @nvshmem_uint64_p(ptr, i64, i32)
declare void @nvshmemx_signal_op(ptr, i64, i32, i32)
declare void @nvshmem_fence()
declare void @nvshmem_quiet()
declare i64 @nvshmem_signal_wait_until(ptr, i32, i64)

define i64 @dsl_nvshmem_putmem_nbi(i64 %dst, i64 %src, i64 %bytes, i32 %pe) #0 {
  %d = inttoptr i64 %dst to ptr
  %s = inttoptr i64 %src to ptr
  call void @nvshmem_putmem_nbi(ptr %d, ptr %s, i64 %bytes, i32 %pe)
  ret i64 0
}

define i64 @dsl_nvshmemx_putmem_nbi_block(i64 %dst, i64 %src, i64 %bytes, i32 %pe) #0 {
  %d = inttoptr i64 %dst to ptr
  %s = inttoptr i64 %src to ptr
  call void @nvshmemx_putmem_nbi_block(ptr %d, ptr %s, i64 %bytes, i32 %pe)
  ret i64 0
}

define i64 @dsl_nvshmem_putmem_signal_nbi(i64 %dst, i64 %src, i64 %bytes, i64 %flag, i64 %value, i32 %op, i32 %pe) #0 {
  %d = inttoptr i64 %dst to ptr
  %s = inttoptr i64 %src to ptr
  %f = inttoptr i64 %flag to ptr
  call void @nvshmem_putmem_signal_nbi(ptr %d, ptr %s, i64 %bytes, ptr %f, i64 %value, i32 %op, i32 %pe)
  ret i64 0
}

define i64 @dsl_nvshmem_uint64_p(i64 %dst, i64 %value, i32 %pe) #0 {
  %d = inttoptr i64 %dst to ptr
  call void @nvshmem_uint64_p(ptr %d, i64 %value, i32 %pe)
  ret i64 0
}

define i64 @dsl_nvshmemx_signal_op(i64 %flag, i64 %value, i32 %op, i32 %pe) #0 {
  %f = inttoptr i64 %flag to ptr
  call void @nvshmemx_signal_op(ptr %f, i64 %value, i32 %op, i32 %pe)
  ret i64 0
}

define i64 @dsl_nvshmem_fence() #0 {
  call void @nvshmem_fence()
  ret i64 0
}

define i64 @dsl_nvshmem_quiet() #0 {
  call void @nvshmem_quiet()
  ret i64 0
}

define i64 @dsl_nvshmem_signal_wait_until(i64 %flag, i32 %cmp, i64 %value) #0 {
  %f = inttoptr i64 %flag to ptr
  %r = call i64 @nvshmem_signal_wait_until(ptr %f, i32 %cmp, i64 %value)
  ret i64 %r
}

attributes #0 = { alwaysinline }
