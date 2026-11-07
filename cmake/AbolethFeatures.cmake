include_guard(GLOBAL)

# Feature registry.
#
# Every registered feature becomes an ABOLETH_HAS_<NAME> preprocessor macro that
# is *always* defined, as 0 or 1. Sources therefore use `#if ABOLETH_HAS_EFA`
# rather than `#ifdef`, and -Wundef catches typos in feature names instead of
# silently compiling the disabled branch.

set(ABOLETH_FEATURE_LIST "" CACHE INTERNAL "Registered Aboleth features")

# Declares ABOLETH_HAS_<NAME>, defaulting to OFF.
macro(aboleth_add_feature NAME)
  set(ABOLETH_HAS_${NAME} OFF CACHE INTERNAL "Whether ${NAME} support is available")
  list(APPEND ABOLETH_FEATURE_LIST ${NAME})
  set(ABOLETH_FEATURE_LIST "${ABOLETH_FEATURE_LIST}" CACHE INTERNAL "")
endmacro()

macro(aboleth_enable_feature NAME)
  if(NOT "${NAME}" IN_LIST ABOLETH_FEATURE_LIST)
    message(FATAL_ERROR
      "ABOLETH_HAS_${NAME} is not a known feature; declare it with aboleth_add_feature first")
  endif()
  set(ABOLETH_HAS_${NAME} ON CACHE INTERNAL "" FORCE)
endmacro()

# Creates the interface target carrying every ABOLETH_HAS_* definition. Call
# once, after all discovery has run — the values are baked in here.
function(aboleth_load_feature_flags)
  add_library(aboleth_feature_flags INTERFACE)
  add_library(aboleth::feature_flags ALIAS aboleth_feature_flags)
  foreach(feature IN LISTS ABOLETH_FEATURE_LIST)
    if(ABOLETH_HAS_${feature})
      target_compile_definitions(aboleth_feature_flags INTERFACE ABOLETH_HAS_${feature}=1)
    else()
      target_compile_definitions(aboleth_feature_flags INTERFACE ABOLETH_HAS_${feature}=0)
    endif()
  endforeach()
endfunction()

# ---------------------------------------------------------------------------

aboleth_add_feature(CUDA)
aboleth_add_feature(HIP)

aboleth_add_feature(VERBS)
aboleth_add_feature(MLX5)
aboleth_add_feature(EFA)
aboleth_add_feature(LIBFABRIC)

aboleth_add_feature(TRITON_PLUGIN)
aboleth_add_feature(NVSHMEM)
aboleth_add_feature(NCCL_GIN)
aboleth_add_feature(PYTHON_BINDINGS)
