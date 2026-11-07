add_subdirectory(compiler)
add_subdirectory(python)

set(ABOLETH_EXPERIMENT_PYTHON "${PROJECT_SOURCE_DIR}/experiments/python")

# aboleth_add_python_test(<name> SCRIPT <file> [ARGS ...] [PYTHON <exe>]
#   plus everything aboleth_add_gated_test takes)
function(aboleth_add_python_test name)
  cmake_parse_arguments(arg "SUDO" "SCRIPT;TIMEOUT;PYTHON" "ARGS;LABELS;NEEDS;ENV" ${ARGN})
  if(NOT arg_PYTHON)
    set(arg_PYTHON "${Python_EXECUTABLE}")
  endif()
  set(_forward)
  if(arg_SUDO)
    list(APPEND _forward SUDO)
  endif()
  if(arg_TIMEOUT)
    list(APPEND _forward TIMEOUT ${arg_TIMEOUT})
  endif()
  aboleth_add_gated_test(${name}
    COMMAND "${arg_PYTHON}" "${arg_SCRIPT}" ${arg_ARGS}
    LABELS ${arg_LABELS}
    NEEDS ${arg_NEEDS}
    ENV ${arg_ENV}
    ${_forward})
endfunction()

# The paper's Listing 2: a fused FP8 dispatch over peer memory and over the
# NIC with the doorbell forwarded by the host, which needs no privilege.
aboleth_add_python_test(hardware.tier2-dispatch-p2p
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_tier2_dispatch.py"
  ARGS --transport p2p
  LABELS hardware NEEDS gpu multi-gpu TIMEOUT 600
  ENV "PYTHONUNBUFFERED=1")
aboleth_add_python_test(hardware.tier2-dispatch-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_tier2_dispatch.py"
  ARGS --transport rdma
  LABELS hardware NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "PYTHONUNBUFFERED=1" "ABOLETH_RDMA_SUBMIT=staged")

# ---------------------------------------------------------------------------
# Binary hygiene
# ---------------------------------------------------------------------------

if(ABOLETH_BUILD_RUNTIME)
  add_test(NAME lint.no-dt-needed
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/lint/check_link_hygiene.py"
            "$<TARGET_FILE:aboleth_rdma_abi>"
            --forbid "^lib(ibverbs|mlx5|efa|fabric|cuda)\\.so")
  set_tests_properties(lint.no-dt-needed PROPERTIES LABELS "lint")
endif()

if(ABOLETH_HAS_TRITON_PLUGIN)
  add_test(NAME lint.plugin-link-closure
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/lint/check_link_hygiene.py"
            "$<TARGET_FILE:AbolethPlugin>"
            --allow-only "^(libtriton\\.so|libstdc\\+\\+\\.so|libgcc_s\\.so|libc\\.so|libm\\.so|ld-linux)")
  set_tests_properties(lint.plugin-link-closure PROPERTIES LABELS "lint")

  add_test(NAME lint.single-libtriton
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/lint/check_single_libtriton.py")
  set_tests_properties(lint.single-libtriton PROPERTIES
    LABELS "lint"
    ENVIRONMENT "${ABOLETH_TEST_ENV}")
endif()

if(NOT ABOLETH_HAS_TRITON_PLUGIN)
  return()
endif()

# ---------------------------------------------------------------------------
# Smokes — no privileges required
# ---------------------------------------------------------------------------

aboleth_add_python_test(smoke.triton
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/triton_smoke.py"
  LABELS smoke NEEDS gpu TIMEOUT 600)

aboleth_add_python_test(smoke.frontend
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_frontend_smoke.py"
  LABELS smoke NEEDS gpu TIMEOUT 600)

aboleth_add_python_test(smoke.kv-cache
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/kv_cache_api_smoke.py"
  LABELS smoke NEEDS gpu multi-gpu TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1")

aboleth_add_python_test(smoke.moe-phase
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/compiled_moe_smoke.py"
  LABELS smoke NEEDS gpu multi-gpu TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1")

aboleth_add_python_test(smoke.pack-timeout
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_pack_timeout.py"
  LABELS smoke TIMEOUT 300)

aboleth_add_python_test(smoke.p2p-triton
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_p2p_triton.py"
  LABELS smoke NEEDS gpu TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1")

if(NOT ABOLETH_BUILD_RUNTIME)
  return()
endif()

aboleth_add_python_test(smoke.proxy-triton
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_proxy_triton.py"
  LABELS smoke NEEDS gpu hca TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1")

aboleth_add_python_test(smoke.proxy-error
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_proxy_error.py"
  LABELS smoke NEEDS gpu hca TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1")

aboleth_add_python_test(smoke.local-proxy
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_local_endpoint_smoke.py"
  ARGS --transport proxy --left-device 0 --right-device 1
  LABELS smoke NEEDS gpu multi-gpu hca TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1")

aboleth_add_python_test(smoke.proxy-concurrent-stress
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_concurrent_producers.py"
  ARGS --producers 512 --transfers 4 --bytes 256 --transport proxy
  LABELS smoke NEEDS gpu hca TIMEOUT 300)

aboleth_add_python_test(smoke.multiprocess-proxy
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_multiprocess_smoke.py"
  ARGS --transport proxy --all-gpus
  LABELS smoke NEEDS gpu gpus=8 hca TIMEOUT 300)

aboleth_add_python_test(smoke.multiprocess-p2p
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_multiprocess_smoke.py"
  ARGS --transport p2p --all-gpus
  LABELS smoke NEEDS gpu gpus=8 TIMEOUT 300)

# ---------------------------------------------------------------------------
# Hardware — needs privileges to map the NIC's user-access region
# ---------------------------------------------------------------------------

# Every SUDO arm pins ABOLETH_RDMA_SUBMIT=direct: these tests exist to prove
# the GPU-rung datapath, and without the pin a host whose driver refuses the
# IO mapping even under sudo would silently pass them via the staged
# fallback. The staged arms below run the same scripts unprivileged.
aboleth_add_python_test(hardware.rdma-abi
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_rdma_abi_test.py"
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.rdma-triton
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_rdma_triton.py"
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=direct")

# The standing proof that the fragment artifact path and the in-kernel path emit
# the same wire bytes. The artifact links whichever device library this build
# emitted, which need not be the running GPU's architecture: the bitcode is
# linked into the client's module and compiled for the client's target.
list(GET ABOLETH_RDMA_BITCODE_ARCHS 0 _rdma_device_arch)
aboleth_add_python_test(hardware.fragment-differential
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_fragment_differential.py"
  LABELS hardware SUDO NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_ARCH=${_rdma_device_arch}"
      "ABOLETH_RDMA_SUBMIT=direct")

# The retirement gate of the generated Tier-1 library: every generated op's
# wire bytes against the hand-written facade's, over identical launchers, with
# a Triton bitcode-consumer arm on top. The hand-written device code may only
# be cut while this is green.
aboleth_add_python_test(hardware.tier1-differential
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_tier1_differential.py"
  LABELS hardware SUDO NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1"
      "ABOLETH_NVCC=${CMAKE_CUDA_COMPILER}" "ABOLETH_RDMA_SUBMIT=direct")

# The standing proof that a register-direct inline payload is the same transfer
# as a staged one, and that the queue accounting counts WQEBBs. Every size the
# granted inline capacity covers is posted, read back on the receiver, and
# checked field by field in the sender's send queue.
aboleth_add_python_test(hardware.inline-differential
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_inline_differential.py"
  LABELS hardware SUDO NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.rdma-pingpong
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_rdma_pingpong_triton.py"
  ARGS --rounds 10000 --transfers 32
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.runtime-smoke
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_runtime_smoke.py"
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.local-rdma
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_local_endpoint_smoke.py"
  ARGS --transport rdma --left-device 0 --right-device 1
  LABELS hardware SUDO NEEDS gpu multi-gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.multiprocess-rdma
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_multiprocess_smoke.py"
  ARGS --transport rdma --all-gpus
  LABELS hardware SUDO NEEDS gpu gpus=8 mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.concurrent-stress
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_concurrent_producers.py"
  ARGS --producers 512 --transfers 4 --bytes 256
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.completion-timeout
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_completion_timeout.py"
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 120
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.deferred-lifecycle
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_deferred_lifecycle.py"
  ARGS --warmup-rounds 2 --rounds 20
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "PYTHONUNBUFFERED=1" "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.moe-exchange
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_moe_exchange.py"
  ARGS --remote-tokens 64 --hidden 32 --warmup-iterations 2 --iterations 10
       --transport rdma
  LABELS hardware SUDO NEEDS gpu mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=direct")

aboleth_add_python_test(hardware.8gpu-exchange
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_8gpu_exchange.py"
  LABELS hardware SUDO NEEDS gpu gpus=8 mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=direct")

# The rank-local arm of the multi-node smoke: two ranks in two processes on
# this host, bootstrapped over TCP into a uniform RDMA World. Stage A only —
# the World, bootstrap, and region-exchange plumbing that the multi-node ladder
# stands on, with no compiled datapath, so this stays a usable regression net.
aboleth_add_python_test(hardware.world-multinode-local
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_world_multinode.py"
  ARGS --local-ranks 2 --stage a
  LABELS hardware SUDO NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "PYTHONUNBUFFERED=1" "ABOLETH_RDMA_SUBMIT=direct")

# ---------------------------------------------------------------------------
# Hardware, staged submission — no privileges needed
# ---------------------------------------------------------------------------
# The same datapath with the one UAR MMIO store moved to the runtime's
# doorbell agent (ABOLETH_RDMA_SUBMIT=staged): the kernel still builds every
# WQE, advances the DBREC, and retires completions, so no IO mapping is taken
# and none of these arms carries SUDO. The three differentials are the
# bitwise proof that staged mode changes no wire byte; the concurrent-stress
# arm is the SQ-wrap stress any queue ordering change must pass.

aboleth_add_python_test(hardware.rdma-abi-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_rdma_abi_test.py"
  LABELS hardware NEEDS gpu mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.rdma-triton-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_rdma_triton.py"
  LABELS hardware NEEDS gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.fragment-differential-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_fragment_differential.py"
  LABELS hardware NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_ARCH=${_rdma_device_arch}"
      "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.inline-differential-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_inline_differential.py"
  LABELS hardware NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.tier1-differential-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_tier1_differential.py"
  LABELS hardware NEEDS gpu multi-gpu mlx5 TIMEOUT 600
  ENV "TRITON_ALWAYS_COMPILE=1"
      "ABOLETH_NVCC=${CMAKE_CUDA_COMPILER}" "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.rdma-pingpong-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_rdma_pingpong_triton.py"
  ARGS --rounds 10000 --transfers 32
  LABELS hardware NEEDS gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.runtime-smoke-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_runtime_smoke.py"
  LABELS hardware NEEDS gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.local-rdma-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_local_endpoint_smoke.py"
  ARGS --transport rdma --left-device 0 --right-device 1
  LABELS hardware NEEDS gpu multi-gpu mlx5 TIMEOUT 300
  ENV "TRITON_ALWAYS_COMPILE=1" "ABOLETH_RDMA_SUBMIT=staged")

aboleth_add_python_test(hardware.concurrent-stress-staged
  SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_concurrent_producers.py"
  ARGS --producers 512 --transfers 4 --bytes 256
  LABELS hardware NEEDS gpu mlx5 TIMEOUT 300
  ENV "ABOLETH_RDMA_SUBMIT=staged")

# The direct-mlx5 probes map the adapter's doorbell page from user space, which
# is what makes most of them root-only. Fields: name|binary|args|SUDO-or-empty.
foreach(probe IN ITEMS
    "wqe-loopback|gpu_wqe_loopback||SUDO"
    "wqe-batch|gpu_wqe_batch|batch,32|SUDO"
    "wqe-per-wqe|gpu_wqe_batch|per-wqe,32|SUDO"
    "backend-p2p|aboleth_backend_test|p2p,32|"
    "backend-rdma|aboleth_backend_test|rdma,32|SUDO"
    "pingpong-p2p|aboleth_pingpong|p2p,10000,32|"
    "pingpong-rdma|aboleth_pingpong|rdma,10000,32|SUDO")
  string(REPLACE "|" ";" _fields "${probe}")
  list(GET _fields 0 _name)
  list(GET _fields 1 _binary)
  list(GET _fields 2 _args)
  list(GET _fields 3 _sudo)
  string(REPLACE "," ";" _args "${_args}")
  if(TARGET ${_binary})
    aboleth_add_gated_test(hardware.${_name}
      COMMAND "$<TARGET_FILE:${_binary}>" ${_args}
      LABELS hardware NEEDS gpu mlx5 TIMEOUT 300 ${_sudo})
  endif()
endforeach()

# ---------------------------------------------------------------------------
# EFA
# ---------------------------------------------------------------------------

if(ABOLETH_HAS_EFA AND TARGET aboleth_efa_smoke)
  # Staged submission: the kernel builds and posts every descriptor and the
  # runtime's doorbell agent moves them into the adapter queue, so no queue
  # memory is mapped for the GPU and no privileges are needed.
  aboleth_add_gated_test(efa.staged-smoke
    COMMAND "$<TARGET_FILE:aboleth_efa_smoke>"
    LABELS "efa;smoke" NEEDS gpu efa TIMEOUT 120
    ENV "ABOLETH_EFA_SUBMIT=host")

  aboleth_add_gated_test(efa.host-smoke
    COMMAND "$<TARGET_FILE:aboleth_efa_smoke>"
    LABELS "efa;hardware" SUDO NEEDS gpu efa TIMEOUT 120
    ENV "ABOLETH_EFA_SUBMIT=gpu")

  # Same smoke with descriptors and doorbells posted by the host through the
  # GPU's queues; passes on platforms whose fabric drops GPU stores to the BAR.
  aboleth_add_gated_test(efa.host-post-smoke
    COMMAND "$<TARGET_FILE:aboleth_efa_smoke>"
    LABELS "efa;hardware" SUDO NEEDS gpu efa TIMEOUT 120
    ENV "ABOLETH_EFA_SUBMIT=gpu;ABOLETH_EFA_HOST_POST=1")

  set(_efa_python "${ABOLETH_EFA_PYTHON}")
  if(NOT _efa_python)
    set(_efa_python "${PROJECT_SOURCE_DIR}/.venv-efa/bin/python")
  endif()
  if(EXISTS "${_efa_python}")
    aboleth_add_python_test(efa.proxy-smoke
      PYTHON "${_efa_python}"
      SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_local_endpoint_smoke.py"
      ARGS --transport proxy --left-device 0 --right-device 1
      LABELS "efa;smoke" NEEDS gpu multi-gpu efa TIMEOUT 120
      ENV "ABOLETH_PROXY_PROVIDER=efa" "FI_EFA_ENABLE_SHM_TRANSFER=0"
          "TRITON_CACHE_DIR=${PROJECT_SOURCE_DIR}/.triton/efa-cache")

    aboleth_add_python_test(efa.proxy-multiprocess
      PYTHON "${_efa_python}"
      SCRIPT "${ABOLETH_EXPERIMENT_PYTHON}/aboleth_multiprocess_smoke.py"
      ARGS --transport proxy --all-gpus
      LABELS "efa;smoke" NEEDS gpu gpus=8 efa TIMEOUT 300
      ENV "ABOLETH_PROXY_PROVIDER=efa" "FI_EFA_ENABLE_SHM_TRANSFER=0"
          "TRITON_CACHE_DIR=${PROJECT_SOURCE_DIR}/.triton/efa-multiprocess-cache")
  endif()
endif()

# ---------------------------------------------------------------------------
# Benchmarks. These measure rather than assert, so they carry the `bench` label
# and stay out of the default suite; scripts/aboleth-bench runs them with other
# parameters.
# ---------------------------------------------------------------------------

set(_bench_presets "rdma-benchmark|")
foreach(entry IN LISTS _bench_presets)
  string(REPLACE "|" ";" _fields "${entry}")
  list(GET _fields 0 _preset)
  list(GET _fields 1 _extra_needs)
  aboleth_add_gated_test(bench.${_preset}
    COMMAND "${PROJECT_SOURCE_DIR}/scripts/aboleth-bench" ${_preset}
    LABELS bench SUDO NEEDS gpu mlx5 ${_extra_needs} TIMEOUT 600
    ENV "ABOLETH_BENCH_NO_SUDO=1")
endforeach()

# Tier-1 microbenchmark smoke: 2 sizes, 5 iterations, correctness only, on
# the aboleth-generic arm. Its default transport is RDMA, so it needs an HCA,
# but not sudo — staged submission is the unprivileged path.
foreach(_tier1 tier1-ladder tier1-alltoall)
  aboleth_add_gated_test(bench.${_tier1}-smoke
    COMMAND "${PROJECT_SOURCE_DIR}/scripts/aboleth-bench" ${_tier1} --smoke
    LABELS "bench;smoke" NEEDS gpu multi-gpu mlx5 TIMEOUT 300
    ENV "ABOLETH_BENCH_NO_SUDO=1")
endforeach()
