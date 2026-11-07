include_guard(GLOBAL)

# pybind11 comes from the build interpreter's site-packages, so the extension is
# always built against the same headers the importing interpreter has.

set(ABOLETH_PYBIND11_FOUND OFF)

if(ABOLETH_ENABLE_PYTHON STREQUAL "OFF")
  return()
endif()

# pybind11_add_module needs the module-build half of the Python package (headers
# and, on some platforms, the import library); the interpreter alone is not
# enough. Development.Module deliberately excludes libpython, which an extension
# must not link.
find_package(Python 3.9 QUIET COMPONENTS Interpreter Development.Module)
if(NOT Python_Development.Module_FOUND)
  message(STATUS "[Aboleth] pybind11: ${Python_EXECUTABLE} has no development headers")
  return()
endif()

if(NOT pybind11_DIR)
  aboleth_python_query(_pybind11_dir "import pybind11; print(pybind11.get_cmake_dir())")
  if(_pybind11_dir)
    set(pybind11_DIR "${_pybind11_dir}")
  endif()
endif()

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  set(ABOLETH_PYBIND11_FOUND ON)
  message(STATUS "[Aboleth] pybind11: ${pybind11_INCLUDE_DIR}")
else()
  message(STATUS "[Aboleth] pybind11: not found in ${Python_EXECUTABLE}")
endif()
