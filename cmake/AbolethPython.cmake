include_guard(GLOBAL)

# One interpreter for the whole build: it resolves the Triton install, provides
# the pybind11 CMake package, and runs every Python-side test. Presets pin
# Python_EXECUTABLE to the repo venv so the extension suffix and the pybind11
# ABI always match the interpreter that will import the result.
if(NOT Python_EXECUTABLE AND EXISTS "${PROJECT_SOURCE_DIR}/.venv/bin/python")
  set(Python_EXECUTABLE "${PROJECT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "" FORCE)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)

# aboleth_python_query(<RESULT_VAR> <python-expression>)
# Evaluates an expression in the build interpreter; leaves RESULT_VAR empty on
# any failure so callers can treat it as "not available".
function(aboleth_python_query RESULT_VAR EXPRESSION)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${EXPRESSION}"
    OUTPUT_VARIABLE _out
    ERROR_QUIET
    RESULT_VARIABLE _status
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_status EQUAL 0)
    set(${RESULT_VAR} "${_out}" PARENT_SCOPE)
  else()
    set(${RESULT_VAR} "" PARENT_SCOPE)
  endif()
endfunction()

aboleth_python_query(ABOLETH_PYTHON_EXT_SUFFIX
  "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

message(STATUS "[Aboleth] Python: ${Python_EXECUTABLE} (${Python_VERSION})")
