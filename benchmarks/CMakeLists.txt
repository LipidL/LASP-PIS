# Model-level benchmark executables.  One executable links exactly one model
# library so both models can coexist without sharing a process.
set(BENCHMARK_TARGETS "")
if(TARGET hpnn::hpnn)
  add_executable(lapis_bench_hpnn src/bench.cu)
  target_link_libraries(lapis_bench_hpnn PRIVATE hpnn::hpnn)
  target_compile_definitions(lapis_bench_hpnn PRIVATE LAPIS_BENCH_HPNN=1)
  set_target_properties(lapis_bench_hpnn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN/hpnn")
  list(APPEND BENCHMARK_TARGETS lapis_bench_hpnn)
endif()
if(TARGET mace::mace)
  add_executable(lapis_bench_mace src/bench.cu)
  target_link_libraries(lapis_bench_mace PRIVATE mace::mace)
  target_compile_definitions(lapis_bench_mace PRIVATE LAPIS_BENCH_MACE=1)
  set_target_properties(lapis_bench_mace PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN/mace")
  list(APPEND BENCHMARK_TARGETS lapis_bench_mace)
endif()
if(NOT BENCHMARK_TARGETS)
  message(FATAL_ERROR "LAPIS_BUILD_BENCHMARKS requires HPNN or MACE targets")
endif()
foreach(target IN LISTS BENCHMARK_TARGETS)
  target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
endforeach()
