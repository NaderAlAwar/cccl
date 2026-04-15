// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

namespace
{
using size_type    = int;
using bitmask_type = std::uint32_t;

constexpr size_type warp_size      = 32;
constexpr size_type mask_word_bits = 32;

_CCCL_HOST_DEVICE constexpr size_type word_index(size_type index)
{
  return index / mask_word_bits;
}

void check_copy_if_else_correctness(
  const thrust::device_vector<int>& output,
  const thrust::device_vector<bitmask_type>& mask_words,
  const thrust::device_vector<size_type>& valid_count,
  const thrust::device_vector<int>& lhs_values,
  const thrust::device_vector<int>& rhs_values,
  const thrust::device_vector<std::uint8_t>& lhs_validity,
  const thrust::device_vector<std::uint8_t>& rhs_validity)
{
  const thrust::host_vector<int> h_output                = output;
  const thrust::host_vector<bitmask_type> h_mask_words   = mask_words;
  const thrust::host_vector<size_type> h_valid_count     = valid_count;
  const thrust::host_vector<int> h_lhs_values            = lhs_values;
  const thrust::host_vector<int> h_rhs_values            = rhs_values;
  const thrust::host_vector<std::uint8_t> h_lhs_validity = lhs_validity;
  const thrust::host_vector<std::uint8_t> h_rhs_validity = rhs_validity;

  size_type expected_valid_count = 0;

  for (std::size_t index = 0; index < h_output.size(); ++index)
  {
    const bool use_lhs        = (index & 1u) == 0u;
    const auto& source_values = use_lhs ? h_lhs_values : h_rhs_values;
    const auto& source_valid  = use_lhs ? h_lhs_validity : h_rhs_validity;

    const bool is_valid      = source_valid[index] != 0;
    const int expected_value = is_valid ? source_values[index] : 0;

    if (h_output[index] != expected_value)
    {
      throw std::runtime_error("custom_copy_if_else correctness check failed: unexpected output value.");
    }

    const std::size_t current_word = index / mask_word_bits;
    const std::uint32_t bit_mask   = std::uint32_t{1} << (index % mask_word_bits);
    const bool bit_set             = (h_mask_words[current_word] & bit_mask) != 0;

    if (bit_set != is_valid)
    {
      throw std::runtime_error("custom_copy_if_else correctness check failed: unexpected validity mask bit.");
    }

    expected_valid_count += is_valid ? 1 : 0;
  }

  if (h_valid_count.size() != 1 || h_valid_count[0] != expected_valid_count)
  {
    throw std::runtime_error("custom_copy_if_else correctness check failed: unexpected valid count.");
  }
}

template <size_type block_size, size_type leader_lane = 0, typename T>
__device__ T single_lane_block_sum_reduce(T lane_value)
{
  static_assert(block_size <= 1024, "Invalid block size.");
  static_assert(cuda::std::is_arithmetic_v<T>, "Invalid non-arithmetic type.");

  constexpr auto warps_per_block = block_size / warp_size;

  const auto lane_id = static_cast<size_type>(threadIdx.x % warp_size);
  const auto warp_id = static_cast<size_type>(threadIdx.x / warp_size);

  __shared__ T lane_values[warp_size];

  if (lane_id == leader_lane)
  {
    lane_values[warp_id] = lane_value;
  }
  __syncthreads();

  T result{0};
  if (warp_id == 0)
  {
    __shared__ typename cub::WarpReduce<T>::TempStorage temp_storage;
    lane_value = (lane_id < warps_per_block) ? lane_values[lane_id] : T{0};
    result     = cub::WarpReduce<T>(temp_storage).Sum(lane_value);
  }

  __syncthreads();
  return result;
}

template <size_type block_size, typename Filter>
__global__ void copy_if_else_kernel(
  const int* lhs_values,
  const int* rhs_values,
  const std::uint8_t* lhs_validity,
  const std::uint8_t* rhs_validity,
  int* output,
  bitmask_type* mask_words,
  size_type size,
  Filter filter,
  size_type* valid_count)
{
  constexpr size_type leader_lane = 0;

  const auto lane_id = static_cast<size_type>(threadIdx.x % warp_size);
  auto item_index    = static_cast<size_type>(blockIdx.x * block_size + threadIdx.x);
  const auto stride  = static_cast<size_type>(gridDim.x * block_size);
  size_type warp_valid_count{0};

  unsigned int active_mask = __ballot_sync(0xFFFF'FFFFu, item_index < size);
  while (item_index < size)
  {
    const bool use_lhs                    = filter(item_index);
    const int* selected_values            = use_lhs ? lhs_values : rhs_values;
    const std::uint8_t* selected_validity = use_lhs ? lhs_validity : rhs_validity;
    const bool is_valid                   = selected_validity[item_index] != 0;

    output[item_index] = is_valid ? selected_values[item_index] : 0;

    const bitmask_type ballot = __ballot_sync(active_mask, is_valid);

    if (lane_id == leader_lane)
    {
      mask_words[word_index(item_index)] = ballot;
      warp_valid_count += __popc(ballot);
    }

    item_index += stride;
    active_mask = __ballot_sync(active_mask, item_index < size);
  }

  const size_type block_count = single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
  if (threadIdx.x == 0)
  {
    atomicAdd(valid_count, block_count);
  }
}

template <typename Filter>
void launch_copy_if_else_kernel(
  const int* lhs_values,
  const int* rhs_values,
  const std::uint8_t* lhs_validity,
  const std::uint8_t* rhs_validity,
  int* output,
  bitmask_type* mask_words,
  size_type size,
  Filter filter,
  size_type* valid_count,
  cudaStream_t stream)
{
  if (size <= 0)
  {
    return;
  }

  constexpr size_type block_size = 256;
  const size_type num_blocks     = (size + block_size - 1) / block_size;

  copy_if_else_kernel<block_size><<<num_blocks, block_size, 0, stream>>>(
    lhs_values, rhs_values, lhs_validity, rhs_validity, output, mask_words, size, filter, valid_count);
}
} // namespace

void custom_copy_if_else(nvbench::state& state)
try
{
  constexpr size_type num_items = 64;
  constexpr size_type num_words = num_items / mask_word_bits;

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
  thrust::device_vector<bitmask_type> mask_words(num_words, thrust::no_init);
  thrust::device_vector<size_type> valid_count(1, thrust::no_init);

  auto* d_lhs_values   = thrust::raw_pointer_cast(lhs_values.data());
  auto* d_rhs_values   = thrust::raw_pointer_cast(rhs_values.data());
  auto* d_lhs_validity = thrust::raw_pointer_cast(lhs_validity.data());
  auto* d_rhs_validity = thrust::raw_pointer_cast(rhs_validity.data());
  auto* d_output       = thrust::raw_pointer_cast(output.data());
  auto* d_mask_words   = thrust::raw_pointer_cast(mask_words.data());
  auto* d_valid_count  = thrust::raw_pointer_cast(valid_count.data());

  state.add_element_count(num_items);
  state.add_global_memory_reads<int>(num_items, "LhsValues");
  state.add_global_memory_reads<int>(num_items, "RhsValues");
  state.add_global_memory_reads<std::uint8_t>(num_items, "LhsValidity");
  state.add_global_memory_reads<std::uint8_t>(num_items, "RhsValidity");
  state.add_global_memory_writes<int>(num_items, "Output");
  state.add_global_memory_writes<bitmask_type>(num_words, "MaskWords");
  state.add_global_memory_writes<size_type>(1, "ValidCount");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const cudaStream_t stream = launch.get_stream().get_stream();

    cudaMemsetAsync(d_valid_count, 0, sizeof(size_type), stream);
    launch_copy_if_else_kernel(
      d_lhs_values,
      d_rhs_values,
      d_lhs_validity,
      d_rhs_validity,
      d_output,
      d_mask_words,
      num_items,
      [] __device__(int index) {
        return (index & 1) == 0;
      },
      d_valid_count,
      stream);
  });

  check_copy_if_else_correctness(output, mask_words, valid_count, lhs_values, rhs_values, lhs_validity, rhs_validity);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(custom_copy_if_else).set_name("custom_copy_if_else");
