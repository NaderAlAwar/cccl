// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_transform.cuh>

#include <cuda/iterator>

#include <nvbench_helper.cuh>

#include <benchmarks/common/generate_input.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>

namespace
{
constexpr cudf::size_type mask_word_bits = sizeof(cudf::bitmask_type) * 8;
} // namespace

template <typename DataType, bool HasNulls>
void run_device_transform_copy_if_else(nvbench::state& state)
try
{
  auto const num_items = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto input_type  = cudf::type_to_id<DataType>();
  auto bool_type   = cudf::type_id::BOOL8;
  auto const input = create_random_table({input_type, input_type, bool_type}, row_count{num_items});

  if constexpr (!HasNulls)
  {
    // Strip null masks so the nulls=0 cases stay good proxies for the no-null custom-kernel path,
    // where cuDF does not load lhs/rhs null information when the inputs are not nullable.
    input->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(1).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(2).set_null_mask(rmm::device_buffer{}, 0);
  }

  cudf::column_view lhs(input->view().column(0));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view decision(input->view().column(2));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if constexpr (HasNulls)
  {
    // Kept for reference, but disabled at the benchmark entry point.
    // Findings:
    // - Packed input masks can still be modeled in a row-wise DeviceTransform proxy.
    // - Packed output masks do not match the row-wise output shape and require a separate
    //   word-wise path, which pushes this away from the simple "pass all arrays to
    //   DeviceTransform" proxy we want.
    auto output = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_to_id<DataType>()},
      num_items,
      cudf::mask_state::UNINITIALIZED,
      cudf::get_default_stream());

    auto output_view          = output->mutable_view();
    auto* d_output            = output_view.template data<DataType>();
    auto* d_output_mask       = output_view.null_mask();
    auto const* d_lhs         = lhs.data<DataType>();
    auto const* d_rhs         = rhs.data<DataType>();
    auto const* d_decision    = decision.data<bool>();
    auto const* lhs_mask      = lhs.null_mask();
    auto const* rhs_mask      = rhs.null_mask();
    auto const* decision_mask = decision.null_mask();
    auto const bitmask_bytes  = cudf::bitmask_allocation_size_bytes(num_items);
    auto const num_mask_words = cudf::num_bitmask_words(num_items);

    state.add_global_memory_reads<int8_t>(num_items * (sizeof(DataType) + 2 * sizeof(bool)) + 3 * bitmask_bytes);
    state.add_global_memory_writes<int8_t>(num_items * sizeof(DataType) + bitmask_bytes);

    auto value_transform_op = [d_lhs, d_rhs, d_decision] __device__(cudf::size_type index) {
      return d_decision[index] ? d_lhs[index] : d_rhs[index];
    };

    auto mask_transform_op =
      [num_items, d_decision, lhs_mask, rhs_mask, decision_mask] __device__(cudf::size_type word_index) {
        cudf::bitmask_type word = 0;
        auto const first_index  = word_index * mask_word_bits;

        for (cudf::size_type bit = 0; bit < mask_word_bits; ++bit)
        {
          auto const index = first_index + bit;
          if (index >= num_items)
          {
            break;
          }

          auto const decision_is_valid = cudf::bit_value_or(decision_mask, index, true);
          auto const choose_lhs        = decision_is_valid && d_decision[index];
          auto const selected_is_valid =
            choose_lhs ? cudf::bit_value_or(lhs_mask, index, true) : cudf::bit_value_or(rhs_mask, index, true);

          if (selected_is_valid)
          {
            word |= (cudf::bitmask_type{1} << bit);
          }
        }

        return word;
      };

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto const launch_stream = launch.get_stream().get_stream();

      timer.start();
      cub::DeviceTransform::Transform(
        cuda::counting_iterator<cudf::size_type>{0}, d_output, num_items, value_transform_op, launch_stream);
      cub::DeviceTransform::Transform(
        cuda::counting_iterator<cudf::size_type>{0}, d_output_mask, num_mask_words, mask_transform_op, launch_stream);
      timer.stop();
    });
  }
  else
  {
    auto output = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_to_id<DataType>()},
      num_items,
      cudf::mask_state::UNALLOCATED,
      cudf::get_default_stream());

    auto* d_output         = output->mutable_view().template data<DataType>();
    auto const* d_lhs      = lhs.data<DataType>();
    auto const* d_rhs      = rhs.data<DataType>();
    auto const* d_decision = decision.data<bool>();

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
  }
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename DataType>
void device_transform_copy_if_else(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const nulls = static_cast<bool>(state.get_int64("nulls"));
  if (nulls)
  {
    state.skip("Skipping: nullable DeviceTransform proxy disabled. Packed mask inputs are viable, but packed mask "
               "outputs do not fit the intended single-pass DeviceTransform proxy shape.");
    return;
  }

  run_device_transform_copy_if_else<DataType, false>(state);
}

using Types = nvbench::type_list<int16_t, uint32_t, double>;

NVBENCH_BENCH_TYPES(device_transform_copy_if_else, NVBENCH_TYPE_AXES(Types))
  .set_name("device_transform_copy_if_else")
  .set_type_axes_names({"DataType"})
  .add_int64_axis("nulls", {true, false})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728});
