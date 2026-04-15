// This was suggested by codex from other benchmarks that use valid_if
// internally since there is no direct benchmark for valid_if. Inspired by
// copy_if_else, replace_nulls, and concatenate benchmarks which use valid_if
// internally.

// Build notes:
// 1. Build and install compatible C++ builds of RMM and cuDF into one prefix:
//    `git clone https://github.com/rapidsai/rmm.git /path/to/rmm`
//    `git clone https://github.com/rapidsai/cudf.git /path/to/cudf`
//    `cmake -S /path/to/rmm/cpp -B /path/to/rmm/cpp/build -DCMAKE_INSTALL_PREFIX=/path/to/rapids-install`
//    `cmake --build /path/to/rmm/cpp/build`
//    `cmake --install /path/to/rmm/cpp/build`
//    `cmake -S /path/to/cudf/cpp -B /path/to/cudf/cpp/build -DCMAKE_INSTALL_PREFIX=/path/to/rapids-install
//    -DCMAKE_PREFIX_PATH=/path/to/rapids-install -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=OFF` `cmake --build
//    /path/to/cudf/cpp/build` `cmake --install /path/to/cudf/cpp/build`
// 2. Configure CCCL with those packages visible to CMake:
//    `cmake -S /path/to/cccl -B /path/to/cccl/build/cub-benchmark --preset cub-benchmark
//    -Dcudf_DIR=/path/to/rapids-install/lib/cmake/cudf -Drmm_DIR=/path/to/rapids-install/lib/cmake/rmm` Alternatively:
//    `CMAKE_PREFIX_PATH=/path/to/rapids-install cmake -S /path/to/cccl -B /path/to/cccl/build/cub-benchmark --preset
//    cub-benchmark`
// 3. Build this target:
//    `cmake --build /path/to/cccl/build/cub-benchmark --target cub.bench.hierarchical.cudf.cudf_valid_if.base`

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <benchmarks/common/cudf_random_input.cuh>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <nvbench/nvbench.cuh>

static void bench_valid_if_threshold(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const true_fraction = state.get_float64("true_fraction");
  auto const threshold     = static_cast<cudf::size_type>(num_rows * true_fraction);

  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + num_rows;

  auto pred = [threshold] __device__(cudf::size_type i) {
    return i < threshold;
  };

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_writes<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_rows));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto [mask, null_count] = cudf::detail::valid_if(begin, end, pred, stream, cudf::get_current_device_resource_ref());
  });
}

NVBENCH_BENCH(bench_valid_if_threshold)
  .set_name("valid_if_threshold")
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728})
  .add_float64_axis("true_fraction", {0.0, 0.3, 0.5, 1.0});

static void bench_valid_if_copy_if_else_shape(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");

  auto const inputs = cub::benchmarks::cudf_input::make_valid_if_copy_if_else_shape_input(num_rows, null_probability);

  auto lhs      = inputs->view().column(0);
  auto rhs      = inputs->view().column(1);
  auto decision = inputs->view().column(2);

  auto stream     = cudf::get_default_stream();
  auto lhs_dv     = cudf::column_device_view::create(lhs, stream);
  auto rhs_dv     = cudf::column_device_view::create(rhs, stream);
  auto decision_d = cudf::column_device_view::create(decision, stream);

  auto lhs_iter = cudf::detail::make_optional_iterator<int32_t>(*lhs_dv, cudf::nullate::DYNAMIC{lhs.nullable()});
  auto rhs_iter = cudf::detail::make_optional_iterator<int32_t>(*rhs_dv, cudf::nullate::DYNAMIC{rhs.nullable()});

  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + num_rows;

  auto pred = [lhs_iter, rhs_iter, decision = *decision_d] __device__(cudf::size_type idx) {
    return decision.element<bool>(idx) ? lhs_iter[idx].has_value() : rhs_iter[idx].has_value();
  };

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_writes<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_rows));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto [mask, null_count] = cudf::detail::valid_if(begin, end, pred, stream, cudf::get_current_device_resource_ref());
  });
}

NVBENCH_BENCH(bench_valid_if_copy_if_else_shape)
  .set_name("valid_if_copy_if_else_shape")
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728})
  .add_float64_axis("null_probability", {0.0, 0.3});

static void bench_valid_if_any_null_across_cols(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const null_probability = state.get_float64("null_probability");

  auto const table = cub::benchmarks::cudf_input::make_valid_if_any_null_input(num_rows, num_cols, null_probability);

  auto stream  = cudf::get_default_stream();
  auto d_table = cudf::table_device_view::create(table->view(), stream);

  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + num_rows;

  auto pred = [d_table = *d_table] __device__(cudf::size_type idx) {
    return !thrust::any_of(thrust::seq, d_table.begin(), d_table.end(), [idx](auto col) {
      return col.is_null(idx);
    });
  };

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_writes<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_rows));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto [mask, null_count] = cudf::detail::valid_if(begin, end, pred, stream, cudf::get_current_device_resource_ref());
  });
}

NVBENCH_BENCH(bench_valid_if_any_null_across_cols)
  .set_name("valid_if_any_null_across_cols")
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152})
  .add_int64_axis("num_cols", {2, 8, 64, 256})
  .add_float64_axis("null_probability", {0.0, 0.3});
