cmake_minimum_required(VERSION 3.24)
project(lapis_workspace LANGUAGES CXX CUDA)
# Keep the compilation database usable by clangd. CMake otherwise routes CUDA
# include directories through nvcc's --options-file, which clangd cannot read.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
include(CTest)
option(LAPIS_BUILD_MACE "Build MACE MH1 inference" ON)
# Ensure the optional benchmark runtime is available before adding consumers.
option(MACE_BUILD_EXPLICIT_REFERENCE "Build MACE with materialized connectivity for benchmarks" OFF)
if(LAPIS_BUILD_MACE AND MACE_BUILD_EXPLICIT_REFERENCE)
  set(LAPIS_BUILD_EXPLICIT_REFERENCE ON CACHE BOOL "Build benchmark runtime" FORCE)
endif()
add_subdirectory(lapis)
if(LAPIS_BUILD_MACE)
  add_subdirectory(chemio)
  add_subdirectory(checkpoint)
  add_subdirectory(mace)
endif()
