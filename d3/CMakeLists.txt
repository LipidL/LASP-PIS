cmake_minimum_required(VERSION 3.24)
project(d3 VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(NOT TARGET lapis::lapis)
  find_package(lapis 0.1.0 EXACT CONFIG REQUIRED)
endif()
add_library(d3 STATIC src/d3.cu)
add_library(d3::d3 ALIAS d3)
target_include_directories(d3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(d3 PUBLIC lapis::lapis)
set_target_properties(d3 PROPERTIES POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(BUILD_TESTING)
  add_executable(d3_tests tests/physics.cpp)
  target_link_libraries(d3_tests PRIVATE d3::d3)
  add_test(NAME d3.physics COMMAND d3_tests)
  set_tests_properties(d3.physics PROPERTIES RESOURCE_LOCK lapis_gpu)
endif()
install(TARGETS d3 EXPORT d3Targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES data/LGPL-3.txt data/GPL-3.txt data/NOTICE.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/d3)
install(EXPORT d3Targets NAMESPACE d3:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/d3)
configure_package_config_file(cmake/d3Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/d3Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/d3)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/d3ConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/d3Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/d3ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/d3)
