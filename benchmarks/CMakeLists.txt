# Model-level benchmark executable for MACE MH1.
if(NOT TARGET mace::mace)
  message(FATAL_ERROR "LAPIS_BUILD_BENCHMARKS requires LAPIS_BUILD_MACE=ON")
endif()
add_executable(lapis_bench_mace src/bench.cu)
target_link_libraries(lapis_bench_mace PRIVATE mace::mace)
target_compile_features(lapis_bench_mace PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(lapis_bench_mace PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN/mace")
