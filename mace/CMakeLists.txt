cmake_minimum_required(VERSION 3.24)
project(mace VERSION 0.1.0 LANGUAGES CXX CUDA)
# Keep the compilation database usable by clangd. CMake otherwise routes CUDA
# include directories through nvcc's --options-file, which clangd cannot read.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CTest)
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET lapis::lapis)
  find_package(lapis 0.1.0 EXACT CONFIG REQUIRED)
endif()

foreach(dependency IN ITEMS chemio checkpoint)
  if(NOT TARGET ${dependency}::${dependency})
    find_package(${dependency} 0.1.0 EXACT CONFIG REQUIRED)
  endif()
endforeach()
add_library(mace SHARED src/mh1.cu src/c_api.cpp src/weights.cpp src/structure.cpp)
add_library(mace::mace ALIAS mace)
target_include_directories(mace PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mace PUBLIC lapis::lapis PRIVATE CUDA::cublas chemio::chemio checkpoint::checkpoint)
target_compile_options(mace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(mace PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON VERSION ${PROJECT_VERSION} SOVERSION 0
  INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH TRUE)
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
  add_library(mace_explicit SHARED src/mh1.cu src/c_api.cpp src/weights.cpp src/structure.cpp)
  target_include_directories(mace_explicit PRIVATE include ${CMAKE_CURRENT_BINARY_DIR}/include)
  target_link_libraries(mace_explicit PRIVATE lapis::explicit CUDA::cublas chemio::chemio checkpoint::checkpoint)
  target_compile_definitions(mace_explicit PRIVATE mace_EXPORTS)
  target_compile_options(mace_explicit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  set_target_properties(mace_explicit PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
endif()

add_executable(mace-main src/main.cpp)
target_link_libraries(mace-main PRIVATE mace::mace)
set_target_properties(mace-main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS mace-main RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(BUILD_TESTING)
  add_executable(mace_weights_dump tests/weights.cpp src/weights.cpp)
  target_include_directories(mace_weights_dump PRIVATE src)
  target_link_libraries(mace_weights_dump PRIVATE checkpoint::checkpoint)
  set(MACE_TEST_CHECKPOINT "" CACHE FILEPATH "Original MH1 checkpoint for native C ABI checks")
  if(MACE_TEST_CHECKPOINT)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME mace.checkpoint_native COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/native.py --library $<TARGET_FILE:mace>
      --archive ${MACE_TEST_CHECKPOINT})
  endif()
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
