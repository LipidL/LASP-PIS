cmake_minimum_required(VERSION 3.24)
project(lapis_workspace LANGUAGES CXX CUDA)
# nvcc 11+ moves CUDA include paths into includes_CUDA.rsp and passes it via
# --options-file, which clangd/clang cannot read (every .cu file would be
# parsed without -I). Modules/Compiler/NVIDIA.cmake re-enables it when the
# CUDA language is enabled, so this must come after project().
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
# Keep build/compile_commands.json available for clangd without -D flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CTest)
option(LAPIS_BUILD_MACE "Build MACE MH1 inference" ON)
option(LAPIS_BUILD_HPNN "Build simplified HPNN inference" ON)
option(LAPIS_BUILD_D3 "Build two-body DFT-D3(BJ) dispersion" ON)
option(LAPIS_BUILD_BENCHMARKS "Build the model-level benchmark harness" OFF)
# Ensure the optional benchmark runtime is available before adding consumers.
option(MACE_BUILD_EXPLICIT_REFERENCE
  "Build MACE with materialized connectivity for benchmarks" OFF)
if(LAPIS_BUILD_MACE AND MACE_BUILD_EXPLICIT_REFERENCE)
  set(LAPIS_BUILD_EXPLICIT_REFERENCE ON CACHE BOOL
    "Build benchmark runtime" FORCE)
endif()
add_subdirectory(lapis)
if(LAPIS_BUILD_D3)
  add_subdirectory(d3)
endif()
if(LAPIS_BUILD_MACE OR LAPIS_BUILD_HPNN)
  add_subdirectory(chemio)
  add_subdirectory(checkpoint)
endif()
if(LAPIS_BUILD_MACE)
  add_subdirectory(mace)
endif()
if(LAPIS_BUILD_HPNN)
  add_subdirectory(hpnn)
endif()
if(LAPIS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(BUILD_TESTING AND LAPIS_BUILD_MACE AND LAPIS_BUILD_HPNN AND UNIX)
  add_executable(coexistence_tests tests/coexistence.cpp)
  target_compile_features(coexistence_tests PRIVATE cxx_std_17)
  target_link_libraries(coexistence_tests PRIVATE
    GTest::gtest_main ${CMAKE_DL_LIBS})
  target_compile_definitions(coexistence_tests PRIVATE
    MACE_LIBRARY_PATH="$<TARGET_FILE:mace>"
    HPNN_LIBRARY_PATH="$<TARGET_FILE:hpnn>"
    MACE_MODEL_PATH="${MACE_TEST_MODEL}"
    HPNN_MODEL_PATH="${HPNN_TEST_MODEL}")
  add_dependencies(coexistence_tests mace hpnn)
  include(GoogleTest)
  gtest_discover_tests(coexistence_tests
    TEST_PREFIX "workspace." DISCOVERY_MODE PRE_TEST
    PROPERTIES
      LABELS "native_coexistence" RESOURCE_LOCK lapis_gpu TIMEOUT 300)
endif()
