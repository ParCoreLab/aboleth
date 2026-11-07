# One ctest per module, so a failure names the file rather than the discovery run.

if(NOT ABOLETH_HAS_TRITON_PLUGIN)
  return()
endif()

file(GLOB _python_tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
foreach(module IN LISTS _python_tests)
  cmake_path(GET module STEM name)
  string(REPLACE "test_" "" name "${name}")
  string(REPLACE "_" "-" name "${name}")
  add_test(NAME python.${name}
    COMMAND "${Python_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}" -p "${module}")
  set_tests_properties(python.${name} PROPERTIES
    LABELS "python"
    ENVIRONMENT "${ABOLETH_TEST_ENV}")
endforeach()
