include_guard(GLOBAL)

# Shared CTest plumbing.
#
# Everything that touches a GPU, an HCA, or root goes through
# scripts/run-gated.sh, which probes its prerequisites and exits 77 when they
# are absent. Combined with SKIP_RETURN_CODE that turns "this machine has no EFA
# adapter" into a Skipped test rather than a red one.

set(ABOLETH_RUN_GATED "${PROJECT_SOURCE_DIR}/scripts/run-gated.sh")
set(ABOLETH_TEST_PYTHONPATH
  "${PROJECT_SOURCE_DIR}/python:${PROJECT_SOURCE_DIR}/third_party/triton/python/triton_kernels")

# Environment shared by every test that loads the compiler or the runtime.
set(ABOLETH_TEST_ENV
  "PYTHONPATH=${ABOLETH_TEST_PYTHONPATH}"
  "ABOLETH_BUILD_DIR=${CMAKE_BINARY_DIR}"
  "TRITON_CACHE_DIR=${PROJECT_SOURCE_DIR}/.triton/cache"
  "TORCHINDUCTOR_CACHE_DIR=${PROJECT_SOURCE_DIR}/.torch-cache"
)
if(ABOLETH_HAS_TRITON_PLUGIN)
  list(APPEND ABOLETH_TEST_ENV
    "ABOLETH_PLUGIN_PATH=$<TARGET_FILE:AbolethPlugin>"
    "TRITON_PLUGIN_PATHS=$<TARGET_FILE:AbolethPlugin>")
endif()

# aboleth_add_ir_test(<name>
#   FILE <mlir>                      relative to test/compiler
#   [PASSES <pass>...]               --pass arguments, in order
#   [CHECK_PREFIX <prefix>]
#   [EXPECT_FAIL]                    the run must exit non-zero; stderr is checked
#   [IDEMPOTENT]                     rerunning the pipeline must be a fixed point
#   [ENV <VAR=value>...])
function(aboleth_add_ir_test name)
  cmake_parse_arguments(arg "EXPECT_FAIL;IDEMPOTENT" "FILE;CHECK_PREFIX" "PASSES;ENV" ${ARGN})

  set(_command
    "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/filecheck_runner.py"
    --runner "${CMAKE_CURRENT_SOURCE_DIR}/run.py"
    --filecheck "${ABOLETH_FILECHECK}"
    --input "${CMAKE_CURRENT_SOURCE_DIR}/${arg_FILE}")
  foreach(pass IN LISTS arg_PASSES)
    list(APPEND _command --pass "${pass}")
  endforeach()
  if(arg_CHECK_PREFIX)
    list(APPEND _command --check-prefix "${arg_CHECK_PREFIX}")
  endif()
  if(arg_EXPECT_FAIL)
    list(APPEND _command --expect-fail)
  endif()
  if(arg_IDEMPOTENT)
    list(APPEND _command --idempotent)
  endif()

  add_test(NAME ${name} COMMAND ${_command})
  set_tests_properties(${name} PROPERTIES
    LABELS "compiler-ir"
    ENVIRONMENT "${ABOLETH_TEST_ENV};${arg_ENV}")
endfunction()

# aboleth_add_gated_test(<name>
#   COMMAND <argv>...
#   [LABELS <label>...] [NEEDS gpu|hca|efa|multi-gpu ...] [SUDO]
#   [TIMEOUT <seconds>] [ENV <VAR=value>...] [WORKING_DIRECTORY <dir>])
function(aboleth_add_gated_test name)
  cmake_parse_arguments(arg "SUDO" "TIMEOUT;WORKING_DIRECTORY" "COMMAND;LABELS;NEEDS;ENV" ${ARGN})

  set(_gate "${ABOLETH_RUN_GATED}")
  foreach(need IN LISTS arg_NEEDS)
    list(APPEND _gate --needs ${need})
  endforeach()
  if(arg_SUDO)
    list(APPEND _gate --sudo)
  endif()
  if(arg_TIMEOUT)
    list(APPEND _gate --timeout ${arg_TIMEOUT})
  endif()

  add_test(NAME ${name} COMMAND ${_gate} -- ${arg_COMMAND})
  set_tests_properties(${name} PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS "${arg_LABELS}"
    ENVIRONMENT "${ABOLETH_TEST_ENV};${arg_ENV}")
  if(arg_TIMEOUT)
    math(EXPR _ctest_timeout "${arg_TIMEOUT} + 60")
    set_tests_properties(${name} PROPERTIES TIMEOUT ${_ctest_timeout})
  endif()
  if(arg_WORKING_DIRECTORY)
    set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY "${arg_WORKING_DIRECTORY}")
  endif()
endfunction()
