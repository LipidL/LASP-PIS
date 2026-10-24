cmake_minimum_required(VERSION 3.24)
project(lapis VERSION 0.1.0 LANGUAGES CXX CUDA)
# Keep the compilation database usable by clangd. CMake otherwise routes CUDA
# include directories through nvcc's --options-file, which clangd cannot read.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
find_package(CUDAToolkit REQUIRED)
option(LAPIS_BUILD_EXAMPLES "Build Lennard-Jones example" ${BUILD_TESTING})
option(LAPIS_BUILD_EXPLICIT_REFERENCE "Build optional materialized-connectivity benchmark runtime" OFF)

function(lapis_runtime target)
  add_library(${target} STATIC ${ARGN})
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(${target} PUBLIC CUDA::cudart)
  target_compile_features(${target} PUBLIC cxx_std_17 cuda_std_17)
  target_compile_options(${target} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endfunction()
lapis_runtime(lapis src/runtime.cu)
add_library(lapis::lapis ALIAS lapis)
install(TARGETS lapis EXPORT lapisTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(LAPIS_BUILD_EXPLICIT_REFERENCE)
  lapis_runtime(lapis_explicit src/runtime.cu src/explicit_reference.cu)
  add_library(lapis::explicit ALIAS lapis_explicit)
  set_target_properties(lapis_explicit PROPERTIES EXPORT_NAME explicit)
  # This changes template traversal as well as runtime preprocessing.
  target_compile_definitions(lapis_explicit PUBLIC LAPIS_EXPLICIT_REFERENCE=1)
  install(TARGETS lapis_explicit EXPORT lapisTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(LAPIS_BUILD_EXAMPLES OR BUILD_TESTING)
  add_library(lapis_lj STATIC examples/lj/lj.cu)
  target_include_directories(lapis_lj PUBLIC examples/lj)
  target_link_libraries(lapis_lj PUBLIC lapis::lapis)
endif()
if(BUILD_TESTING)
  add_executable(lapis_tests tests/framework.cu)
  target_link_libraries(lapis_tests PRIVATE lapis_lj)
  add_test(NAME lapis.framework COMMAND lapis_tests)
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT lapisTargets NAMESPACE lapis:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lapis)
configure_package_config_file(cmake/lapisConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/lapisConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lapis)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/lapisConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lapisConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/lapisConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lapis)
