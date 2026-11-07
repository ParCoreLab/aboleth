include_guard(GLOBAL)

# Installation and the AbolethConfig package.
#
# What is installed is what an out-of-tree consumer needs: the runtime library
# and its C ABI header, the plugin, and the device libraries. The compiler's
# static archives are deliberately not exported — they are inputs to the plugin,
# not a supported linking surface, and half of them carry MLIR's ABI.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ABOLETH_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/Aboleth")

if(ABOLETH_BUILD_RUNTIME)
  install(TARGETS aboleth_rdma_abi
    EXPORT AbolethTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/aboleth"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
endif()

if(ABOLETH_HAS_TRITON_PLUGIN)
  # A MODULE library is dlopened, never linked, so it is installed as a plain
  # file rather than exported as a target.
  install(TARGETS AbolethPlugin
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

if(ABOLETH_HAS_EFA)
  foreach(arch IN LISTS ABOLETH_EFA_BITCODE_ARCHS)
    install(FILES "${ABOLETH_LIB_DIR}/aboleth_efa_${arch}.bc"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endforeach()
endif()

if(ABOLETH_BUILD_RUNTIME)
  # Tier 1 links the device library either as bitcode or through the archive;
  # the archive is absent when the build found no llc or nvcc.
  foreach(arch IN LISTS ABOLETH_RDMA_BITCODE_ARCHS)
    install(FILES "${ABOLETH_LIB_DIR}/aboleth_rdma_${arch}.bc"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(FILES "${ABOLETH_LIB_DIR}/libaboleth_device_${arch}.a"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}" OPTIONAL)
  endforeach()
endif()

if(ABOLETH_BUILD_RUNTIME)
  install(EXPORT AbolethTargets
    FILE AbolethTargets.cmake
    NAMESPACE Aboleth::
    DESTINATION "${ABOLETH_INSTALL_CMAKEDIR}")

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/AbolethConfig.cmake.in"
    "@PACKAGE_INIT@\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/AbolethTargets.cmake\")\n")
  configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AbolethConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AbolethConfig.cmake"
    INSTALL_DESTINATION "${ABOLETH_INSTALL_CMAKEDIR}")
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AbolethConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/AbolethConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/AbolethConfigVersion.cmake"
    DESTINATION "${ABOLETH_INSTALL_CMAKEDIR}")
endif()
