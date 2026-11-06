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
# Ensure the optional benchmark runtime is available before adding consumers.
option(MACE_BUILD_EXPLICIT_REFERENCE
  "Build MACE with materialized connectivity for benchmarks" OFF)
if(LAPIS_BUILD_MACE AND MACE_BUILD_EXPLICIT_REFERENCE)
  set(LAPIS_BUILD_EXPLICIT_REFERENCE ON CACHE BOOL
    "Build benchmark runtime" FORCE)
endif()
add_subdirectory(lapis)
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
if(BUILD_TESTING AND LAPIS_BUILD_MACE AND LAPIS_BUILD_HPNN
   AND MACE_TEST_ARCHIVE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  foreach(first IN ITEMS mace hpnn)
    add_test(NAME workspace.${first}_first COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/coexistence.py
      --mace $<TARGET_FILE:mace> --hpnn $<TARGET_FILE:hpnn>
      --archive ${MACE_TEST_ARCHIVE} --first ${first})
  endforeach()
endif()
