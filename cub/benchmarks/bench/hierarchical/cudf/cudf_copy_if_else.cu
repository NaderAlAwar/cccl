/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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
//    `cmake --build /path/to/cccl/build/cub-benchmark --target cub.bench.hierarchical.cudf.cudf_copy_if_else.base`

#include <benchmarks/common/cudf_random_input.cuh>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <nvbench/nvbench.cuh>
#include <rmm/device_buffer.hpp>

template <typename DataType>
static void bench_copy_if_else(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const nulls    = static_cast<bool>(state.get_int64("nulls"));

  auto const input = cub::benchmarks::cudf_input::make_copy_if_else_input<DataType>(num_rows, nulls);

  if (!nulls)
  {
    input->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(1).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(2).set_null_mask(rmm::device_buffer{}, 0);
  }

  cudf::column_view lhs(input->view().column(0));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view decision(input->view().column(2));

  auto const bytes_read    = num_rows * (sizeof(DataType) + sizeof(bool));
  auto const bytes_written = num_rows * sizeof(DataType);
  auto const null_bytes    = nulls ? 2 * cudf::bitmask_allocation_size_bytes(num_rows) : 0;

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(bytes_read);
  state.add_global_memory_writes<int8_t>(bytes_written + null_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto output = cudf::copy_if_else(lhs, rhs, decision);
  });
}

using Types = nvbench::type_list<int16_t, uint32_t, double>;

NVBENCH_BENCH_TYPES(bench_copy_if_else, NVBENCH_TYPE_AXES(Types))
  .set_name("copy_if_else")
  .set_type_axes_names({"DataType"})
  .add_int64_axis("nulls", {true, false})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728});
