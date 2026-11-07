include_guard(GLOBAL)

# Turns the dependency probes into the final set of buildable components. Kept
# separate from the deps/ modules so the "did we find it" and "do we want it"
# decisions do not interleave.

# --- verbs / mlx5 ----------------------------------------------------------

set(_verbs_available OFF)
if(TARGET rdma_headers AND (ABOLETH_HAS_CUDA OR ABOLETH_HAS_HIP))
  set(_verbs_available ON)
endif()
aboleth_resolve_tristate(ABOLETH_ENABLE_VERBS _enable_verbs _verbs_available
  "rdma-core headers or the CUDA toolkit are missing")
if(_enable_verbs)
  aboleth_enable_feature(VERBS)
  if(ABOLETH_HAS_CUDA AND EXISTS "${ABOLETH_RDMA_INCLUDE_DIR}/infiniband/mlx5dv.h")
    aboleth_enable_feature(MLX5)
  endif()
endif()

# --- libfabric / EFA -------------------------------------------------------

if(TARGET libfabric_headers)
  aboleth_enable_feature(LIBFABRIC)
endif()

set(ABOLETH_EFA_DP_DIR "${PROJECT_SOURCE_DIR}/third_party/efa-dp-direct/CUDA")
set(_efa_missing "")
if(NOT ABOLETH_HAS_CUDA)
  string(APPEND _efa_missing " the CUDA toolkit;")
endif()
if(NOT ABOLETH_LIBFABRIC_HAS_EFA)
  string(APPEND _efa_missing " a libfabric built with the EFA provider (rdma/fi_ext_efa.h);")
endif()
if(NOT ABOLETH_HAVE_EFADV)
  string(APPEND _efa_missing " infiniband/efadv.h;")
endif()
if(NOT EXISTS "${ABOLETH_EFA_DP_DIR}/device")
  string(APPEND _efa_missing " the third_party/efa-dp-direct submodule;")
endif()

set(_efa_available OFF)
if(NOT _efa_missing)
  set(_efa_available ON)
endif()
aboleth_resolve_tristate(ABOLETH_ENABLE_EFA _enable_efa _efa_available
  "missing${_efa_missing}")
if(_enable_efa)
  aboleth_enable_feature(EFA)
endif()

# --- runtime ---------------------------------------------------------------

# The runtime's device discovery and memory registration are verbs-based no
# matter which provider is compiled in, so verbs headers are the whole floor.
# The EFA-OFI proxy inside the runtime translation unit is compiled behind
# ABOLETH_HAS_EFA, so libfabric is a requirement of the EFA feature only.
set(_runtime_available OFF)
if((ABOLETH_HAS_CUDA OR ABOLETH_HAS_HIP) AND TARGET rdma_headers)
  set(_runtime_available ON)
endif()
aboleth_resolve_tristate(ABOLETH_ENABLE_RUNTIME ABOLETH_BUILD_RUNTIME _runtime_available
  "the CUDA toolkit or the rdma-core headers are missing")

# --- experiments -----------------------------------------------------------

aboleth_resolve_tristate(ABOLETH_ENABLE_EXPERIMENTS ABOLETH_BUILD_EXPERIMENTS
  ABOLETH_HAS_CUDA "no CUDA toolkit")

# --- python bindings -------------------------------------------------------

set(_python_available OFF)
if(ABOLETH_PYBIND11_FOUND AND ABOLETH_BUILD_RUNTIME)
  set(_python_available ON)
endif()
aboleth_resolve_tristate(ABOLETH_ENABLE_PYTHON _enable_python _python_available
  "pybind11 is not importable from ${Python_EXECUTABLE}, or the runtime is disabled")
if(_enable_python)
  aboleth_enable_feature(PYTHON_BINDINGS)
endif()
