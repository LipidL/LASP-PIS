cmake_minimum_required(VERSION 3.24)
project(mace VERSION 0.1.0 LANGUAGES CXX CUDA)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CTest)
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET lapis::lapis)
  find_package(lapis 0.1.0 EXACT CONFIG REQUIRED)
endif()

add_library(mace SHARED src/mh1.cu src/c_api.cpp)
add_library(mace::mace ALIAS mace)
target_include_directories(mace PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mace PUBLIC lapis::lapis PRIVATE CUDA::cublas)
target_compile_options(mace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(mace PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON VERSION ${PROJECT_VERSION} SOVERSION 0)
generate_export_header(mace EXPORT_FILE_NAME include/mace/export.h)
install(TARGETS mace EXPORT maceTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/mace/export.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mace)
install(EXPORT maceTargets NAMESPACE mace:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mace)
configure_package_config_file(cmake/maceConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/maceConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mace)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/maceConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/maceConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/maceConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mace)

option(MACE_BUILD_EXPLICIT_REFERENCE "Build MACE with materialized connectivity for benchmarks" OFF)
if(MACE_BUILD_EXPLICIT_REFERENCE)
  if(NOT TARGET lapis::explicit)
    message(FATAL_ERROR "Build/install Lapis with LAPIS_BUILD_EXPLICIT_REFERENCE=ON first")
  endif()
  add_library(mace_explicit SHARED src/mh1.cu src/c_api.cpp)
  target_include_directories(mace_explicit PRIVATE include ${CMAKE_CURRENT_BINARY_DIR}/include)
  target_link_libraries(mace_explicit PRIVATE lapis::explicit CUDA::cublas)
  target_compile_definitions(mace_explicit PRIVATE mace_EXPORTS)
  target_compile_options(mace_explicit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  set_target_properties(mace_explicit PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
endif()

set(MACE_TEST_ARCHIVE "" CACHE FILEPATH "MH1 archive for native C ABI checks")
if(BUILD_TESTING AND MACE_TEST_ARCHIVE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME mace.native COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/native.py
    --library $<TARGET_FILE:mace> --archive ${MACE_TEST_ARCHIVE})
  if(TARGET mace_explicit)
    add_test(NAME mace.explicit COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/native.py
      --library $<TARGET_FILE:mace_explicit> --archive ${MACE_TEST_ARCHIVE})
  endif()
endif()
