// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include <benchmarks/common/generate_input.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>

namespace
{
template <typename DataType>
void check_copy_if_else_correctness(
  const cudf::column_view& output,
  const cudf::column_view& lhs,
  const cudf::column_view& rhs,
  const cudf::column_view& decision,
  rmm::cuda_stream_view stream)
{
  auto const* output_data   = output.data<DataType>();
  auto const* lhs_data      = lhs.data<DataType>();
  auto const* rhs_data      = rhs.data<DataType>();
  auto const* decision_data = decision.data<bool>();

  bool const values_match = thrust::all_of(
    thrust::cuda::par.on(stream.value()),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(output.size()),
    [output_data, lhs_data, rhs_data, decision_data] __device__(cudf::size_type index) {
      auto const expected = decision_data[index] ? lhs_data[index] : rhs_data[index];
      return output_data[index] == expected;
    });

  if (!values_match)
  {
    throw std::runtime_error("device_transform_copy_if_else_conditional_load correctness check failed: unexpected "
                             "output value.");
  }
}
} // namespace

template <typename DataType>
void device_transform_copy_if_else_conditional_load(nvbench::state& state, nvbench::type_list<DataType>)
try
{
  auto const num_items = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto input_type  = cudf::type_to_id<DataType>();
  auto bool_type   = cudf::type_id::BOOL8;
  auto const input = create_random_table({input_type, input_type, bool_type}, row_count{num_items});

  input->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
  input->get_column(1).set_null_mask(rmm::device_buffer{}, 0);
  input->get_column(2).set_null_mask(rmm::device_buffer{}, 0);

  cudf::column_view lhs(input->view().column(0));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view decision(input->view().column(2));

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<DataType>()}, num_items, cudf::mask_state::UNALLOCATED, cudf::get_default_stream());

  auto* d_output         = output->mutable_view().template data<DataType>();
  auto const* d_lhs      = lhs.data<DataType>();
  auto const* d_rhs      = rhs.data<DataType>();
  auto const* d_decision = decision.data<bool>();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(num_items * (sizeof(DataType) + sizeof(bool)));
  state.add_global_memory_writes<int8_t>(num_items * sizeof(DataType));

  auto transform_op = [d_lhs, d_rhs, d_decision] __device__(cudf::size_type index) {
    return d_decision[index] ? d_lhs[index] : d_rhs[index];
  };

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    auto const launch_stream = launch.get_stream().get_stream();

    timer.start();
    cub::DeviceTransform::Transform(
      cuda::counting_iterator<cudf::size_type>{0}, d_output, num_items, transform_op, launch_stream);
    timer.stop();
  });

  check_copy_if_else_correctness<DataType>(output->view(), lhs, rhs, decision, stream);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using Types = nvbench::type_list<int16_t, uint32_t, double>;

NVBENCH_BENCH_TYPES(device_transform_copy_if_else_conditional_load, NVBENCH_TYPE_AXES(Types))
  .set_name("device_transform_copy_if_else_conditional_load")
  .set_type_axes_names({"DataType"})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728});
