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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include <benchmarks/common/cudf_random_input.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/default_stream.hpp>

namespace
{
constexpr int mask_word_bits = 32;

void check_copy_if_else_correctness(
  const cudf::column_view& output, int valid_count, const cudf::column_view& expected, rmm::cuda_stream_view stream)
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

void hierarchical_copy_if_else(nvbench::state& state)
try
{
  constexpr cudf::size_type num_items = 64;
  constexpr int num_words             = (num_items + mask_word_bits - 1) / mask_word_bits;

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

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<int>()}, num_items, cudf::mask_state::UNINITIALIZED, stream);
  auto output_view = output->mutable_view();
  cudf::detail::device_scalar<cudf::size_type> valid_count{0, stream, cudf::get_current_device_resource_ref()};

  auto* d_output      = output_view.data<int>();
  auto* d_mask_words  = output_view.null_mask();
  auto* d_valid_count = valid_count.data();

  auto indices       = cuda::counting_iterator<cudf::size_type>{0};
  auto filter_values = cuda::make_transform_iterator(
    indices, [decision = *decision_d, has_nulls = decision.has_nulls()] __device__(cudf::size_type index) -> bool {
      return (!has_nulls || decision.is_valid_nocheck(index)) && decision.element<bool>(index);
    });
  auto output_iterator =
    cuda::make_transform_output_iterator(d_output, [] __device__(cuda::std::optional<int> result) -> int {
      return result.value_or(0);
    });
  auto transform_op = [] __device__(auto item) -> cuda::std::optional<int> {
    const auto [lhs_value, rhs_value, select_lhs] = item;
    return select_lhs ? lhs_value : rhs_value;
  };
  auto epilog_op =
    [d_mask_words,
     d_valid_count] __device__(auto group, cuda::std::int64_t word_index, cuda::std::optional<int> result) {
      const std::uint32_t word = cuda::device::ballot(group, result.has_value());

      if (cuda::gpu_thread.is_root_rank(group))
      {
        d_mask_words[word_index] = word;

        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_valid_count(*d_valid_count);
        atomic_valid_count.fetch_add(__popc(word), cuda::memory_order_relaxed);
      }
    };

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

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    const auto launch_stream = launch.get_stream().get_stream();
    valid_count.set_value_to_zero_async(stream);

    timer.start();

    cub::DeviceSegmentedTransform::TransformEpilog(
      cuda::zip_iterator{lhs_iter, rhs_iter, filter_values},
      output_iterator,
      num_words,
      mask_word_bits,
      transform_op,
      epilog_op,
      launch_stream);

    timer.stop();
  });

  auto const host_valid_count = valid_count.value(stream);
  output->set_null_count(num_items - host_valid_count);
  auto expected = cudf::copy_if_else(lhs, rhs, decision);
  check_copy_if_else_correctness(output->view(), host_valid_count, expected->view(), stream);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(hierarchical_copy_if_else).set_name("hierarchical_copy_if_else");
