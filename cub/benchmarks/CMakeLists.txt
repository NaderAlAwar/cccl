include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

cccl_get_nvbench_helper()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

set(
  CUDF_ROOT
  ""
  CACHE PATH
  "Path to installed cuDF package root for CUB benchmarks."
)
set(
  RMM_ROOT
  ""
  CACHE PATH
  "Path to installed RMM package root for CUB benchmarks."
)
set(
  FLASHINFER_ROOT
  ""
  CACHE PATH
  "Path to FlashInfer repo root for CUB benchmarks."
)
set(
  TRTLLM_ROOT
  ""
  CACHE PATH
  "Path to TensorRT-LLM repo root for CUB benchmarks."
)
set(
  TRTLLM_LIB_DIR
  ""
  CACHE PATH
  "Path to TensorRT-LLM installed libraries for CUB benchmarks."
)
set(
  TRTLLM_TENSORRT_INCLUDE_DIR
  "/usr/local/tensorrt/include"
  CACHE PATH
  "Path to TensorRT headers for TensorRT-LLM CUB benchmarks."
)

if (TRTLLM_ROOT OR TRTLLM_LIB_DIR)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(rmm CONFIG QUIET HINTS "${RMM_ROOT}")
find_package(cudf CONFIG QUIET HINTS "${CUDF_ROOT}")

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  set(message_type FATAL_ERROR)
  if (CCCL_ENABLE_CLANG_TIDY)
    # We are here because CI has force-enabled clang-tidy. We must use a debug build for
    # this because certain clang-tidy checks (such as out of bounds or clang static
    # analyzer) work better when they see assert()'s. In this case we don't actually
    # intend to run any of the benchmarks, we just need them to be compilable, so a simple
    # warning is enough.
    #
    # We don't ignore this outright (by making it say, DEBUG or VERBOSE), because it's
    # possible that a user may accidentally stumble into enabling the option.
    set(message_type WARNING)
  endif()
  message(${message_type} "CUB benchmarks must be built in release mode.")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(
    FATAL_ERROR
    "CMAKE_CUDA_ARCHITECTURES must be set to build CUB benchmarks."
  )
endif()

set(benches_meta_target cub.all.benches)
add_custom_target(${benches_meta_target})

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(
    GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach (test_dir IN LISTS contents)
    if (IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

create_benchmark_registry()

function(get_bench_ranges src bench_name)
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%RANGE%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL "${param_regex}" matches "${file_data}")

  set(ranges "")

  foreach (match IN LISTS matches)
    string(REGEX MATCH "${param_regex}" unused "${match}")

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(range ${CMAKE_MATCH_3})
    set(ranges "${ranges}${def}|${label}=${range},")

    string(REPLACE ":" ";" range "${range}")
    list(LENGTH range range_len)

    if (NOT "${range_len}" STREQUAL 3)
      message(FATAL_ERROR "Range should be represented as 'start:end:step'")
    endif()
  endforeach()

  string(LENGTH "${ranges}" ranges_length)
  math(EXPR last_character_index "${ranges_length} - 1")
  string(SUBSTRING "${ranges}" 0 ${last_character_index} ranges)
  register_cccl_tuning("${bench_name}" "${ranges}")
endfunction()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)
  get_filename_component(bench_basename "${bench_src}" NAME_WLE)
  file(RELATIVE_PATH bench_relpath "${benches_root}" "${bench_src}")

  set(bench_dialect 17)
  if (bench_relpath MATCHES "^bench/hierarchical/")
    set(bench_dialect 20)
  endif()

  cccl_add_executable(
    ${bench_target}
    NO_METATARGETS
    DIALECT ${bench_dialect}
    SOURCES "${bench_src}"
  )
  target_link_libraries(
    ${bench_target}
    PRIVATE #
      CCCL::CCCL
      cccl.nvbench_helper
      nvbench::main
  )
  if (bench_basename MATCHES "^flashinfer_")
    target_include_directories(
      ${bench_target}
      PRIVATE "${FLASHINFER_ROOT}/include"
    )
    target_compile_options(
      ${bench_target}
      PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    )
  endif()

  if (bench_basename MATCHES "trtllm")
    target_link_libraries(
      ${bench_target}
      PRIVATE CUDA::toolkit "${TRTLLM_LIB_DIR}/libtensorrt_llm.so"
    )
    target_include_directories(
      ${bench_target}
      PRIVATE
        "${TRTLLM_ROOT}/cpp/include"
        "${TRTLLM_ROOT}/cpp"
        "${TRTLLM_TENSORRT_INCLUDE_DIR}"
    )
    target_compile_definitions(${bench_target} PRIVATE ENABLE_BF16 ENABLE_FP8)
    target_compile_options(
      ${bench_target}
      PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    )
  endif()

  if (bench_basename MATCHES "^cudf_")
    target_sources(
      ${bench_target}
      PRIVATE "${benches_root}/common/cudf_random_input.cu"
    )
    target_include_directories(
      ${bench_target}
      PRIVATE "${CMAKE_SOURCE_DIR}/cub"
    )
    target_link_libraries(${bench_target} PRIVATE cudf::cudf rmm::rmm)
    set_target_properties(
      ${bench_target}
      PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:cudf::cudf>;$<TARGET_FILE_DIR:rmm::rmm>"
        INSTALL_RPATH
          "$<TARGET_FILE_DIR:cudf::cudf>;$<TARGET_FILE_DIR:rmm::rmm>"
    )
    target_link_options(
      ${bench_target}
      PRIVATE "$<$<LINK_LANGUAGE:CUDA>:-Wl,--disable-new-dtags>"
    )
  endif()
endfunction()

function(should_add_bench bench_src result)
  get_filename_component(bench_basename "${bench_src}" NAME_WLE)

  set(add_bench_target ON)

  if (bench_basename MATCHES "^flashinfer_")
    if (NOT FLASHINFER_ROOT)
      message(STATUS "Skipping ${bench_basename}: FLASHINFER_ROOT is not set.")
      set(add_bench_target OFF)
    endif()
  endif()

  if (bench_basename MATCHES "trtllm")
    if (NOT TRTLLM_ROOT OR NOT TRTLLM_LIB_DIR)
      message(
        STATUS
        "Skipping ${bench_basename}: TRTLLM_ROOT and TRTLLM_LIB_DIR must be set."
      )
      set(add_bench_target OFF)
    endif()
  endif()

  if (bench_basename MATCHES "^cudf_")
    if (NOT TARGET cudf::cudf OR NOT TARGET rmm::rmm)
      message(
        STATUS
        "Skipping ${bench_basename}: installed cudf and rmm CMake packages were not found."
      )
      set(add_bench_target OFF)
    endif()
  endif()

  set(${result} ${add_bench_target} PARENT_SCOPE)
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach (bench_src IN LISTS bench_srcs)
    should_add_bench("${bench_src}" add_this_bench)
    if (NOT add_this_bench)
      continue()
    endif()

    # base tuning
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "cub.${bench_prefix}.")

    set(base_bench_name "${bench_name}.base")
    add_bench(base_bench_target ${base_bench_name} "${bench_src}")
    add_dependencies(${benches_meta_target} ${base_bench_target})
    target_compile_definitions(${base_bench_target} PRIVATE TUNE_BASE=1)
    target_compile_options(
      ${base_bench_target}
      PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
    )

    if (CUB_ENABLE_TUNING)
      # tuning
      set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${bench_src}"
      )
      get_bench_ranges("${bench_src}" "${bench_name}")
      set(tuning_name "${bench_name}.variant")
      set(tuning_path "${CMAKE_BINARY_DIR}/${tuning_name}.h")
      add_bench(bench_target ${tuning_name} "${bench_src}")
      # for convenience, make tuning variant buildable by default
      file(WRITE "${tuning_path}" "#pragma once\n#define TUNE_BASE 1\n")
      target_compile_options(
        ${bench_target}
        PRIVATE #
          "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
          "-include${tuning_path}"
      )
    else()
      # benchmarking
      register_cccl_benchmark("${bench_name}" "")
    endif()
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach (subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
