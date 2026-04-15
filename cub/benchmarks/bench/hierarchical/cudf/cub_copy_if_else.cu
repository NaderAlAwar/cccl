// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <stdexcept>

#include <nvbench_helper.cuh>

namespace
{
constexpr int mask_word_bits = 32;

struct select_lhs_op
{
  __device__ bool operator()(int index) const
  {
    return (index & 1) == 0;
  }
};

struct select_value_op
{
  const int* lhs_values;
  const int* rhs_values;
  const std::uint8_t* lhs_validity;
  const std::uint8_t* rhs_validity;

  __device__ int operator()(int index) const
  {
    const bool use_lhs        = select_lhs_op{}(index);
    const int* values         = use_lhs ? lhs_values : rhs_values;
    const std::uint8_t* valid = use_lhs ? lhs_validity : rhs_validity;

    return valid[index] != 0 ? values[index] : 0;
  }
};

struct pack_selected_validity_bit_op
{
  const std::uint8_t* lhs_validity;
  const std::uint8_t* rhs_validity;

  __device__ std::uint32_t operator()(int index) const
  {
    const bool use_lhs        = select_lhs_op{}(index);
    const std::uint8_t* valid = use_lhs ? lhs_validity : rhs_validity;

    return valid[index] != 0 ? (std::uint32_t{1} << (index % mask_word_bits)) : 0u;
  }
};

struct popcount_op
{
  __device__ int operator()(std::uint32_t word) const
  {
    return __popc(word);
  }
};

void check_copy_if_else_correctness(
  const thrust::device_vector<int>& output,
  const thrust::device_vector<std::uint32_t>& mask_words,
  const thrust::device_vector<int>& valid_count,
  const thrust::device_vector<int>& lhs_values,
  const thrust::device_vector<int>& rhs_values,
  const thrust::device_vector<std::uint8_t>& lhs_validity,
  const thrust::device_vector<std::uint8_t>& rhs_validity)
{
  const thrust::host_vector<int> h_output                = output;
  const thrust::host_vector<std::uint32_t> h_mask_words  = mask_words;
  const thrust::host_vector<int> h_valid_count           = valid_count;
  const thrust::host_vector<int> h_lhs_values            = lhs_values;
  const thrust::host_vector<int> h_rhs_values            = rhs_values;
  const thrust::host_vector<std::uint8_t> h_lhs_validity = lhs_validity;
  const thrust::host_vector<std::uint8_t> h_rhs_validity = rhs_validity;

  int expected_valid_count = 0;

  for (std::size_t index = 0; index < h_output.size(); ++index)
  {
    const bool use_lhs        = (index & 1u) == 0u;
    const auto& source_values = use_lhs ? h_lhs_values : h_rhs_values;
    const auto& source_valid  = use_lhs ? h_lhs_validity : h_rhs_validity;

    const bool is_valid      = source_valid[index] != 0;
    const int expected_value = is_valid ? source_values[index] : 0;

    if (h_output[index] != expected_value)
    {
      throw std::runtime_error("copy_if_else correctness check failed: unexpected output value.");
    }

    const std::size_t word_index = index / mask_word_bits;
    const std::uint32_t bit_mask = std::uint32_t{1} << (index % mask_word_bits);
    const bool bit_set           = (h_mask_words[word_index] & bit_mask) != 0;

    if (bit_set != is_valid)
    {
      throw std::runtime_error("copy_if_else correctness check failed: unexpected validity mask bit.");
    }

    expected_valid_count += is_valid ? 1 : 0;
  }

  if (h_valid_count.size() != 1 || h_valid_count[0] != expected_valid_count)
  {
    throw std::runtime_error("copy_if_else correctness check failed: unexpected valid count.");
  }
}
} // namespace

void cub_copy_if_else(nvbench::state& state)
try
{
  constexpr int num_items = 64;
  constexpr int num_words = num_items / mask_word_bits;

  thrust::device_vector<int> lhs_values(num_items);
  thrust::device_vector<int> rhs_values(num_items);
  thrust::sequence(lhs_values.begin(), lhs_values.end(), 1000);
  thrust::sequence(rhs_values.begin(), rhs_values.end(), 2000);

  thrust::host_vector<std::uint8_t> h_lhs_validity(num_items);
  thrust::host_vector<std::uint8_t> h_rhs_validity(num_items);

  for (int index = 0; index < num_items; ++index)
  {
    h_lhs_validity[index] = (index % 3) != 0 ? 1u : 0u;
    h_rhs_validity[index] = (index % 5) != 0 ? 1u : 0u;
  }

  thrust::device_vector<std::uint8_t> lhs_validity = h_lhs_validity;
  thrust::device_vector<std::uint8_t> rhs_validity = h_rhs_validity;

  thrust::device_vector<int> output(num_items, thrust::no_init);
  thrust::device_vector<std::uint32_t> mask_words(num_words, thrust::no_init);
  thrust::device_vector<int> valid_count(1, thrust::no_init);

  auto* d_lhs_values   = thrust::raw_pointer_cast(lhs_values.data());
  auto* d_rhs_values   = thrust::raw_pointer_cast(rhs_values.data());
  auto* d_lhs_validity = thrust::raw_pointer_cast(lhs_validity.data());
  auto* d_rhs_validity = thrust::raw_pointer_cast(rhs_validity.data());
  auto* d_output       = thrust::raw_pointer_cast(output.data());
  auto* d_mask_words   = thrust::raw_pointer_cast(mask_words.data());
  auto* d_valid_count  = thrust::raw_pointer_cast(valid_count.data());

  auto indices = cuda::counting_iterator<int>{0};
  auto packed_selected_validity_bits =
    cuda::make_transform_iterator(indices, pack_selected_validity_bit_op{d_lhs_validity, d_rhs_validity});

  std::size_t segmented_reduce_temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
    nullptr,
    segmented_reduce_temp_storage_bytes,
    packed_selected_validity_bits,
    d_mask_words,
    num_words,
    mask_word_bits,
    cuda::std::bit_or<>{},
    std::uint32_t{0});

  thrust::device_vector<nvbench::uint8_t> segmented_reduce_temp_storage(
    segmented_reduce_temp_storage_bytes, thrust::no_init);
  auto* d_segmented_reduce_temp_storage = thrust::raw_pointer_cast(segmented_reduce_temp_storage.data());

  std::size_t reduce_temp_storage_bytes = 0;
  cub::DeviceReduce::TransformReduce(
    nullptr, reduce_temp_storage_bytes, d_mask_words, d_valid_count, num_words, cuda::std::plus<>{}, popcount_op{}, 0);

  thrust::device_vector<nvbench::uint8_t> reduce_temp_storage(reduce_temp_storage_bytes, thrust::no_init);
  auto* d_reduce_temp_storage = thrust::raw_pointer_cast(reduce_temp_storage.data());

  state.add_element_count(num_items);
  state.add_global_memory_reads<int>(num_items, "LhsValues");
  state.add_global_memory_reads<int>(num_items, "RhsValues");
  state.add_global_memory_reads<std::uint8_t>(num_items, "LhsValidity");
  state.add_global_memory_reads<std::uint8_t>(num_items, "RhsValidity");
  state.add_global_memory_reads<std::uint32_t>(num_words, "MaskWords");
  state.add_global_memory_writes<int>(num_items, "Output");
  state.add_global_memory_writes<std::uint32_t>(num_words, "MaskWords");
  state.add_global_memory_writes<int>(1, "ValidCount");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const cudaStream_t stream = launch.get_stream().get_stream();

    cub::DeviceTransform::Transform(
      indices, d_output, num_items, select_value_op{d_lhs_values, d_rhs_values, d_lhs_validity, d_rhs_validity}, stream);

    cub::DeviceSegmentedReduce::Reduce(
      d_segmented_reduce_temp_storage,
      segmented_reduce_temp_storage_bytes,
      packed_selected_validity_bits,
      d_mask_words,
      num_words,
      mask_word_bits,
      cuda::std::bit_or<>{},
      std::uint32_t{0},
      stream);

    cub::DeviceReduce::TransformReduce(
      d_reduce_temp_storage,
      reduce_temp_storage_bytes,
      d_mask_words,
      d_valid_count,
      num_words,
      cuda::std::plus<>{},
      popcount_op{},
      0,
      stream);
  });

  check_copy_if_else_correctness(output, mask_words, valid_count, lhs_values, rhs_values, lhs_validity, rhs_validity);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(cub_copy_if_else).set_name("cub_copy_if_else");
