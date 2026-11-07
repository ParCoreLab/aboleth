include_guard(GLOBAL)

# Language standard, build type, warnings, output layout, and the target
# archetypes every Aboleth target is created through.

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The Makefile-era build left CMAKE_BUILD_TYPE empty, which shipped an
# unoptimised plugin. Default to RelWithDebInfo so `cmake -S . -B build` with no
# further flags produces something usable.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

# One tree: build/lib holds every library and the device bitcode, build/bin the
# executables, build/deps anything acquired rather than authored.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(ABOLETH_LIB_DIR "${CMAKE_BINARY_DIR}/lib")
set(ABOLETH_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
set(ABOLETH_DEPS_DIR "${CMAKE_BINARY_DIR}/deps")

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
# -Wundef is load-bearing: feature macros are always defined as 0/1, so a
# misspelled ABOLETH_HAS_* in an #if is a warning rather than a silent OFF.
set(ABOLETH_CXX_WARNINGS -Wall -Wextra -Wundef)
set(ABOLETH_CUDA_WARNINGS -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wundef)

set(ABOLETH_CXX_FLAGS_DEBUG -O0 -g3 -fno-omit-frame-pointer)
set(ABOLETH_CXX_FLAGS_RELEASE -O3)
set(ABOLETH_CXX_FLAGS_RELWITHDEBINFO -O2 -g -fno-omit-frame-pointer)

# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------

function(aboleth_apply_compile_options target)
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    return()
  endif()

  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:C,CXX>:${ABOLETH_CXX_WARNINGS}>"
    "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Debug>>:${ABOLETH_CXX_FLAGS_DEBUG}>"
    "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release>>:${ABOLETH_CXX_FLAGS_RELEASE}>"
    "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:RelWithDebInfo>>:${ABOLETH_CXX_FLAGS_RELWITHDEBINFO}>"
  )
  if(ABOLETH_HAS_CUDA)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:${ABOLETH_CUDA_WARNINGS}>"
      "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;--generate-line-info>"
      "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>"
      "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O2;--generate-line-info>"
    )
  endif()

  target_compile_definitions(${target} PRIVATE
    $<$<CONFIG:Debug>:ABOLETH_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
  )

  if(NOT TARGET aboleth_feature_flags)
    message(FATAL_ERROR
      "aboleth_feature_flags is missing; aboleth_load_feature_flags() must run before add_subdirectory()")
  endif()
  target_link_libraries(${target} PRIVATE aboleth_feature_flags)
endfunction()

# aboleth_add_library(<name> [STATIC|SHARED|MODULE|OBJECT] <sources>...)
function(aboleth_add_library name)
  add_library(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  aboleth_apply_compile_options(${name})
endfunction()

# aboleth_add_executable(<name> <sources>...)
function(aboleth_add_executable name)
  add_executable(${name} ${ARGN})
  aboleth_apply_compile_options(${name})
endfunction()

# CUDA executables in experiments/ are developer tools: they link their
# transports directly rather than through the runtime's dlopen shims.
function(aboleth_add_cuda_executable name)
  add_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  aboleth_apply_compile_options(${name})
endfunction()

# MLIR and libtriton are built without RTTI or exceptions; anything that links
# them or derives from their types has to match, or vtables and type_infos go
# out of sync at load time.
function(aboleth_use_mlir_abi target)
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
endfunction()
