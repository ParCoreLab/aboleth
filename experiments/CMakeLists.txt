# Developer probes and comparison benchmarks.
#
# Unlike the runtime these link their transports directly: they are diagnostic
# tools run by hand on a machine that is known to have the hardware, so a
# missing libibverbs should be a link error, not a runtime degradation.

set(_verbs_probes
  gpu_wqe_loopback
  gpu_wqe_batch
  aboleth_backend_test
  aboleth_pingpong
)
# The runtime is linked for its device, GID and port selection exports alone
# (aboleth/runtime_c.h, consumed through cuda/aboleth_probe_fabric.h): a probe
# that answered "which HCA, LID or GRH, which MTU" for itself would be a second
# implementation of the machine's addressing, and was. This is a DT_NEEDED on
# the probe binaries, not on libaboleth_rdma_abi.so, so the link hygiene the
# lint.no-dt-needed test enforces on that library is untouched.
if(ABOLETH_HAS_VERBS AND ABOLETH_HAS_MLX5 AND TARGET aboleth_rdma_abi)
  foreach(probe IN LISTS _verbs_probes)
    aboleth_add_cuda_executable(${probe} cuda/${probe}.cu)
    target_link_libraries(${probe} PRIVATE
      aboleth_rdma_abi rdma::ibverbs rdma::mlx5)
  endforeach()
endif()

if(ABOLETH_HAS_EFA)
  # Exercises the GPU-initiated EFA path with no compiler involved: two GPUs,
  # one endpoint each, a kernel that posts a write and a signal through the same
  # device primitives the compiler emits calls to.
  aboleth_add_cuda_executable(aboleth_efa_smoke cuda/aboleth_efa_smoke.cu)
  target_include_directories(aboleth_efa_smoke PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(aboleth_efa_smoke SYSTEM PRIVATE
    "${ABOLETH_EFA_DP_DIR}/common"
    "${ABOLETH_EFA_DP_DIR}/device"
    "${ABOLETH_EFA_DP_DIR}/host")
  target_link_libraries(aboleth_efa_smoke PRIVATE
    aboleth_rdma_abi rdma::ibverbs rdma::efa CUDA::cuda_driver)
endif()

# Tier-1 benchmark arms (experiments/bench): identical templated kernels per
# arm, differing only in the comm adapter they instantiate.
set(_bench_include_dirs
  "${PROJECT_SOURCE_DIR}/experiments/bench"
  "${PROJECT_SOURCE_DIR}/experiments/bench/kernels")
if(ABOLETH_HAS_VERBS AND ABOLETH_HAS_MLX5 AND TARGET aboleth_rdma_abi)
  aboleth_add_cuda_executable(bench_tier1_aboleth bench/bench_tier1_aboleth.cu)
  target_include_directories(bench_tier1_aboleth PRIVATE ${_bench_include_dirs})
  target_link_libraries(bench_tier1_aboleth PRIVATE
    aboleth_rdma_abi rdma::ibverbs rdma::mlx5)
endif()

if(ABOLETH_HAS_NVSHMEM)
  aboleth_add_cuda_executable(bench_tier1_nvshmem bench/bench_tier1_nvshmem.cu)
  set_target_properties(bench_tier1_nvshmem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(bench_tier1_nvshmem PRIVATE NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
  target_include_directories(bench_tier1_nvshmem PRIVATE ${_bench_include_dirs})
  target_link_libraries(bench_tier1_nvshmem PRIVATE
    nvshmem::host nvshmem::device rdma_headers)
endif()

if(ABOLETH_HAS_NCCL_GIN)
  aboleth_add_cuda_executable(bench_tier1_nccl bench/bench_tier1_nccl.cu)
  target_include_directories(bench_tier1_nccl PRIVATE ${_bench_include_dirs})
  target_link_libraries(bench_tier1_nccl PRIVATE nccl::nccl)
  # NCCL's LSA copy (ncclLsaReduceSumLsaCopy) takes device lambdas.
  target_compile_options(bench_tier1_nccl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

  # Host-API reference arm (grouped send/recv, no GIN requirement); built
  # under the same gate because it shares the GIN-capable NCCL install.
  aboleth_add_cuda_executable(bench_tier1_nccl_host
    bench/bench_tier1_nccl_host.cu)
  target_include_directories(bench_tier1_nccl_host
    PRIVATE ${_bench_include_dirs})
  target_link_libraries(bench_tier1_nccl_host PRIVATE nccl::nccl)
endif()

# The Tier-1 benchmark arms compile their device code at -O3, as the
# runtime shape compiler does for the Aboleth arms (common.py), whatever
# the tree's build type.
foreach(bench bench_tier1_aboleth bench_tier1_nvshmem bench_tier1_nccl
        bench_tier1_nccl_host)
  if(TARGET ${bench})
    target_compile_options(${bench} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  endif()
endforeach()
