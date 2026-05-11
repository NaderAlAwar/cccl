// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_hierarchical_transform.cuh>

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
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include <benchmarks/common/generate_input.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>

namespace
{
constexpr int mask_word_bits = 32;

template <typename DataType>
void check_copy_if_else_correctness(
  const cudf::column_view& output, int valid_count, const cudf::column_view& expected, rmm::cuda_stream_view stream)
{
  const auto num_words = static_cast<std::size_t>((output.size() + mask_word_bits - 1) / mask_word_bits);

  const bool values_match = thrust::all_of(
    thrust::cuda::par.on(stream.value()),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(output.size()),
    [output_data   = output.data<DataType>(),
     expected_data = expected.data<DataType>(),
     expected_mask = expected.null_mask()] __device__(cudf::size_type index) {
      const bool is_valid =
        expected_mask == nullptr || ((expected_mask[index / mask_word_bits] >> (index % mask_word_bits)) & 1u) != 0;
      return !is_valid || output_data[index] == expected_data[index];
    });

  if (!values_match)
  {
    throw std::runtime_error("hierarchical_copy_if_else correctness check failed: unexpected output value.");
  }

  const auto* output_mask   = output.null_mask();
  const auto* expected_mask = expected.null_mask();
  const bool masks_match =
    output_mask == nullptr && expected_mask == nullptr
    || (output_mask != nullptr && expected_mask != nullptr
        && thrust::equal(thrust::cuda::par.on(stream.value()),
                         thrust::device_pointer_cast(output_mask),
                         thrust::device_pointer_cast(output_mask) + num_words,
                         thrust::device_pointer_cast(expected_mask)));

  if (!masks_match)
  {
    throw std::runtime_error("hierarchical_copy_if_else correctness check failed: unexpected validity mask.");
  }

  const int expected_valid_count = static_cast<int>(expected.size() - expected.null_count());
  if (valid_count != expected_valid_count)
  {
    throw std::runtime_error("hierarchical_copy_if_else correctness check failed: unexpected valid count.");
  }
}
} // namespace

template <typename DataType, bool HasNulls>
void run_hierarchical_copy_if_else(nvbench::state& state)
try
{
  auto const num_items = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_words = static_cast<int>((num_items + mask_word_bits - 1) / mask_word_bits);

  auto input_type  = cudf::type_to_id<DataType>();
  auto bool_type   = cudf::type_id::BOOL8;
  auto const input = create_random_table({input_type, input_type, bool_type}, row_count{num_items});

  if constexpr (!HasNulls)
  {
    input->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(1).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(2).set_null_mask(rmm::device_buffer{}, 0);
  }

  cudf::column_view lhs(input->view().column(0));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view decision(input->view().column(2));

  auto stream = cudf::get_default_stream();

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<DataType>()},
    num_items,
    HasNulls ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream);
  auto output_view = output->mutable_view();
  cudf::detail::device_scalar<cudf::size_type> valid_count{0, stream, cudf::get_current_device_resource_ref()};

  auto* d_output      = output_view.template data<DataType>();
  auto* d_mask_words  = output_view.null_mask();
  auto* d_valid_count = valid_count.data();

  auto nullable_transform_op =
    [lhs_mask = lhs.null_mask(), rhs_mask = rhs.null_mask(), decision_mask = decision.null_mask()] __device__(
      int index, DataType lhs_value, DataType rhs_value, bool decision_value) -> cuda::std::optional<DataType> {
    auto is_valid = [] __device__(const cudf::bitmask_type* mask, int row) {
      return mask == nullptr || ((mask[row / mask_word_bits] >> (row % mask_word_bits)) & 1u) != 0;
    };

    const bool select_lhs = is_valid(decision_mask, index) && decision_value;
    if (select_lhs)
    {
      if (is_valid(lhs_mask, index))
      {
        return cuda::std::optional<DataType>{lhs_value};
      }

      return cuda::std::nullopt;
    }

    if (is_valid(rhs_mask, index))
    {
      return cuda::std::optional<DataType>{rhs_value};
    }

    return cuda::std::nullopt;
  };
  auto non_null_transform_op = [] __device__(DataType lhs_value, DataType rhs_value, bool select_lhs) -> DataType {
    return select_lhs ? lhs_value : rhs_value;
  };
  auto epilog_op =
    [d_mask_words, d_valid_count] __device__(auto block_group, const auto& results, const auto& indices) {
      if constexpr (HasNulls)
      {
        using results_t                = cuda::std::remove_reference_t<decltype(results)>;
        constexpr int items_per_thread = cuda::std::extent_v<results_t>;
        static_assert(items_per_thread > 0, "Results array must be non-empty.");
        static_assert(mask_word_bits % items_per_thread == 0, "ItemsPerThread must evenly divide the mask word size.");

        constexpr int subgroup_size = mask_word_bits / items_per_thread;
        const int lane_rank         = static_cast<int>(threadIdx.x % mask_word_bits);
        const int subgroup_rank     = lane_rank / subgroup_size;
        const int subgroup_lane     = lane_rank % subgroup_size;

        const cuda::std::uint32_t subgroup_mask =
          items_per_thread == 1 ? 0xffffffffu : ((1u << subgroup_size) - 1u) << (subgroup_rank * subgroup_size);

        cuda::std::uint32_t local_mask         = 0;
        cuda::std::int64_t subgroup_word_index = -1;

        for (int item = 0; item < items_per_thread; ++item)
        {
          const auto& result = results[item];
          const auto index   = indices[item];

          if (index >= 0)
          {
            subgroup_word_index = index / mask_word_bits;
          }

          if (index >= 0 && result.has_value())
          {
            local_mask |= 1u << (subgroup_lane * items_per_thread + item);
          }
        }

        const cuda::std::uint32_t word = __reduce_or_sync(subgroup_mask, local_mask);
        int warp_valid                 = 0;

        if (subgroup_lane == 0 && subgroup_word_index >= 0)
        {
          d_mask_words[subgroup_word_index] = word;
          warp_valid                        = __popc(word);
        }

        const int block_valid = cuda::coop::reduce(block_group, warp_valid, cuda::std::plus<>{});

        if (cuda::gpu_thread.is_root_rank(block_group))
        {
          cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> atomic_valid_count(*d_valid_count);
          atomic_valid_count.fetch_add(static_cast<cudf::size_type>(block_valid), cuda::memory_order_relaxed);
        }
      }
    };

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  {
    auto const bitmask_bytes = cudf::bitmask_allocation_size_bytes(num_items);
    auto const bytes_read    = num_items * (2 * sizeof(DataType) + sizeof(bool)) + (HasNulls ? 3 * bitmask_bytes : 0);
    auto const bytes_written = num_items * sizeof(DataType);
    auto const null_bytes    = HasNulls ? bitmask_bytes : 0;
    state.add_global_memory_reads<int8_t>(bytes_read);
    state.add_global_memory_writes<int8_t>(bytes_written + null_bytes);
  }

  if constexpr (HasNulls)
  {
    auto* lhs_values      = lhs.data<DataType>();
    auto* rhs_values      = rhs.data<DataType>();
    auto* decision_values = decision.data<bool>();
    auto output_iterator =
      cuda::make_transform_output_iterator(d_output, [] __device__(cuda::std::optional<DataType> result) -> DataType {
        return result.value_or(DataType{});
      });

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      const auto launch_stream = launch.get_stream().get_stream();
      valid_count.set_value_to_zero_async(stream);

      timer.start();

      cub::DeviceSegmentedTransform::TransformEpilog(
        cuda::std::make_tuple(lhs_values, rhs_values, decision_values),
        output_iterator,
        num_words,
        mask_word_bits,
        nullable_transform_op,
        epilog_op,
        launch_stream);

      timer.stop();
    });
  }
  else
  {
    auto* lhs_values      = lhs.data<DataType>();
    auto* rhs_values      = rhs.data<DataType>();
    auto* decision_values = decision.data<bool>();

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      const auto launch_stream = launch.get_stream().get_stream();

      timer.start();

      cub::DeviceSegmentedTransform::TransformEpilog(
        cuda::std::make_tuple(lhs_values, rhs_values, decision_values),
        d_output,
        num_words,
        mask_word_bits,
        non_null_transform_op,
        launch_stream);

      timer.stop();
    });
  }

  auto const host_valid_count = HasNulls ? valid_count.value(stream) : num_items;
  output->set_null_count(num_items - host_valid_count);
  auto expected = cudf::copy_if_else(lhs, rhs, decision);
  check_copy_if_else_correctness<DataType>(output->view(), static_cast<int>(host_valid_count), expected->view(), stream);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename DataType>
void hierarchical_copy_if_else(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const nulls = static_cast<bool>(state.get_int64("nulls"));
  if (nulls)
  {
    run_hierarchical_copy_if_else<DataType, true>(state);
  }
  else
  {
    run_hierarchical_copy_if_else<DataType, false>(state);
  }
}

using Types = nvbench::type_list<int16_t, uint32_t, double>;

NVBENCH_BENCH_TYPES(hierarchical_copy_if_else, NVBENCH_TYPE_AXES(Types))
  .set_name("hierarchical_copy_if_else")
  .set_type_axes_names({"DataType"})
  .add_int64_axis("nulls", {true, false})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728});
