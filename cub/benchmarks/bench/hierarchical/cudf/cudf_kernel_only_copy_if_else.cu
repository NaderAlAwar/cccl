/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cub/warp/warp_reduce.cuh>

#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <cuda/std/iterator>
#include <cuda/std/optional>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include <benchmarks/common/cudf_random_input.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace
{
constexpr int mask_word_bits = 32;

template <cudf::size_type block_size, typename T, typename LeftIter, typename RightIter, typename Filter, bool has_nulls>
__launch_bounds__(block_size) __global__ void kernel_only_copy_if_else_kernel(
  LeftIter lhs,
  RightIter rhs,
  Filter filter,
  cudf::mutable_column_device_view out,
  cudf::size_type* __restrict__ const valid_count)
{
  auto tidx = cudf::detail::grid_1d::global_thread_id<block_size>();

  auto const stride         = cudf::detail::grid_1d::grid_stride<block_size>();
  auto const warp_id        = tidx / cudf::detail::warp_size;
  auto const warps_per_grid = stride / cudf::detail::warp_size;

  cudf::size_type const begin      = 0;
  cudf::size_type const end        = out.size();
  cudf::size_type const warp_begin = cudf::word_index(begin);
  cudf::size_type const warp_end   = cudf::word_index(end - 1);

  constexpr cudf::size_type leader_lane{0};
  auto const lane_id = threadIdx.x % cudf::detail::warp_size;

  cudf::size_type warp_valid_count{0};

  cudf::size_type warp_cur = warp_begin + warp_id;
  while (warp_cur <= warp_end)
  {
    auto const index     = static_cast<cudf::size_type>(tidx);
    auto const opt_value = (index < end) ? (filter(index) ? lhs[index] : rhs[index]) : cuda::std::nullopt;
    if (opt_value)
    {
      out.element<T>(index) = static_cast<T>(*opt_value);
    }

    if constexpr (has_nulls)
    {
      int warp_mask = __ballot_sync(0xFFFF'FFFFu, opt_value.has_value());
      if (lane_id == leader_lane)
      {
        out.set_mask_word(warp_cur, warp_mask);
        warp_valid_count += __popc(warp_mask);
      }
    }

    warp_cur += warps_per_grid;
    tidx += stride;
  }

  if constexpr (has_nulls)
  {
    cudf::size_type block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
    if (threadIdx.x == 0)
    {
      atomicAdd(valid_count, block_valid_count);
    }
  }
}

void check_copy_if_else_correctness(
  const cudf::column_view& output, const cudf::column_view& expected, rmm::cuda_stream_view stream)
{
  const auto num_words = static_cast<std::size_t>((output.size() + mask_word_bits - 1) / mask_word_bits);

  const bool values_match = thrust::all_of(
    thrust::cuda::par.on(stream.value()),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(output.size()),
    [output_data   = output.data<int>(),
     expected_data = expected.data<int>(),
     expected_mask = expected.null_mask()] __device__(cudf::size_type index) {
      const bool is_valid =
        expected_mask == nullptr || ((expected_mask[index / mask_word_bits] >> (index % mask_word_bits)) & 1u) != 0;
      return !is_valid || output_data[index] == expected_data[index];
    });

  if (!values_match)
  {
    throw std::runtime_error("cudf_kernel_only_copy_if_else correctness check failed: unexpected output value.");
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
    throw std::runtime_error("cudf_kernel_only_copy_if_else correctness check failed: unexpected validity mask.");
  }

  if (output.null_count() != expected.null_count())
  {
    throw std::runtime_error("cudf_kernel_only_copy_if_else correctness check failed: unexpected valid count.");
  }
}
} // namespace

void cudf_kernel_only_copy_if_else(nvbench::state& state)
try
{
  constexpr cudf::size_type num_items = 64;
  constexpr int num_words             = (num_items + mask_word_bits - 1) / mask_word_bits;
  constexpr int block_size            = 256;

  auto input = cub::benchmarks::cudf_input::make_copy_if_else_input<int>(num_items, true);
  cudf::column_view lhs(input->view().column(0));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view decision(input->view().column(2));

  auto stream     = cudf::get_default_stream();
  auto lhs_dv     = cudf::column_device_view::create(lhs, stream);
  auto rhs_dv     = cudf::column_device_view::create(rhs, stream);
  auto decision_d = cudf::column_device_view::create(decision, stream);

  auto lhs_iter = cudf::detail::make_optional_iterator<int>(*lhs_dv, cudf::nullate::DYNAMIC{lhs.nullable()});
  auto rhs_iter = cudf::detail::make_optional_iterator<int>(*rhs_dv, cudf::nullate::DYNAMIC{rhs.nullable()});

  auto filter = [decision = *decision_d, has_nulls = decision.has_nulls()] __device__(cudf::size_type index) -> bool {
    return (!has_nulls || decision.is_valid_nocheck(index)) && decision.element<bool>(index);
  };

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<int>()}, num_items, cudf::mask_state::UNINITIALIZED, stream);
  auto out_v = cudf::mutable_column_device_view::create(*output, stream);
  cudf::detail::device_scalar<cudf::size_type> valid_count{0, stream, cudf::get_current_device_resource_ref()};

  cudf::size_type num_els = cudf::util::round_up_safe(num_items, cudf::detail::warp_size);
  cudf::detail::grid_1d grid{num_els, block_size, 1};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(num_items);
  state.add_global_memory_reads<int>(num_items, "LhsValues");
  state.add_global_memory_reads<int>(num_items, "RhsValues");
  state.add_global_memory_reads<bool>(num_items, "Decision");
  state.add_global_memory_reads<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_items), "LhsValidity");
  state.add_global_memory_reads<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_items), "RhsValidity");
  state.add_global_memory_writes<int>(num_items, "Output");
  state.add_global_memory_writes<std::uint32_t>(num_words, "MaskWords");
  state.add_global_memory_writes<int>(1, "ValidCount");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    cudaMemsetAsync(valid_count.data(), 0, sizeof(cudf::size_type), stream.value());
    kernel_only_copy_if_else_kernel<block_size, int, decltype(lhs_iter), decltype(rhs_iter), decltype(filter), true>
      <<<grid.num_blocks, block_size, 0, stream.value()>>>(lhs_iter, rhs_iter, filter, *out_v, valid_count.data());
  });

  output->set_null_count(num_items - valid_count.value(stream));
  auto expected = cudf::copy_if_else(lhs, rhs, decision);
  check_copy_if_else_correctness(output->view(), expected->view(), stream);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(cudf_kernel_only_copy_if_else).set_name("cudf_kernel_only_copy_if_else");
