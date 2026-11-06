include_guard(GLOBAL)
option(LAPIS_TEST_USE_SYSTEM_GTEST
  "Use an installed GoogleTest instead of building the pinned test dependency" OFF)
if(LAPIS_TEST_USE_SYSTEM_GTEST)
  find_package(GTest 1.14 CONFIG REQUIRED GLOBAL)
endif()
if(NOT TARGET GTest::gtest_main)
  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

function(add_potential_tests target model library weights)
  if(NOT EXISTS "${weights}")
    message(FATAL_ERROR "${model} test model does not exist: ${weights}. Set ${model}_TEST_MODEL or BUILD_TESTING=OFF.")
  endif()
  set(common "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  # Re-discover parameterized tests when fixtures are added or edited.
  file(GLOB fixtures CONFIGURE_DEPENDS "${common}/structures/*.arc")
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${target}_fixtures.txt"
    CONTENT "${fixtures}\n")
  add_executable(${target} tests/physics.cpp "${common}/physics.cpp" ${fixtures})
  target_include_directories(${target} PRIVATE "${common}")
  target_link_libraries(${target} PRIVATE ${library} chemio::chemio GTest::gtest_main)
  target_compile_definitions(${target} PRIVATE
    TEST_MODEL_PATH="${weights}" TEST_STRUCTURE_DIR="${common}/structures")
  # The manifest changes even when a fixture is removed or an old file is added.
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
    "${fixtures}" "${CMAKE_CURRENT_BINARY_DIR}/${target}_fixtures.txt")
  gtest_discover_tests(${target} TEST_PREFIX "${target}." DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "native_${model}" RESOURCE_LOCK lapis_gpu TIMEOUT 300)
endfunction()
