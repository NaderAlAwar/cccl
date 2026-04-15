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

void check_valid_if_correctness(const thrust::device_vector<bitmask_type>& mask_words,
                                const thrust::device_vector<size_type>& valid_count)
{
  const thrust::host_vector<bitmask_type> h_mask_words = mask_words;
  const thrust::host_vector<size_type> h_valid_count   = valid_count;

  constexpr bitmask_type expected_word = 0x55555555u;
  constexpr size_type expected_valid   = 32;

  if (h_mask_words.size() != 2)
  {
    throw std::runtime_error("custom_valid_if correctness check failed: unexpected number of mask words.");
  }

  for (std::size_t word = 0; word < h_mask_words.size(); ++word)
  {
    if (h_mask_words[word] != expected_word)
    {
      throw std::runtime_error("custom_valid_if correctness check failed: unexpected bitmask word.");
    }
  }

  if (h_valid_count.size() != 1 || h_valid_count[0] != expected_valid)
  {
    throw std::runtime_error("custom_valid_if correctness check failed: unexpected valid count.");
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

template <size_type block_size, typename InputIterator, typename Predicate>
__global__ void
valid_if_kernel(bitmask_type* output, InputIterator begin, size_type size, Predicate predicate, size_type* valid_count)
{
  constexpr size_type leader_lane = 0;

  const auto lane_id = static_cast<size_type>(threadIdx.x % warp_size);
  auto item_index    = static_cast<size_type>(blockIdx.x * block_size + threadIdx.x);
  const auto stride  = static_cast<size_type>(gridDim.x * block_size);
  size_type warp_valid_count{0};

  unsigned int active_mask = __ballot_sync(0xFFFF'FFFFu, item_index < size);
  while (item_index < size)
  {
    const bitmask_type ballot = __ballot_sync(active_mask, predicate(*(begin + item_index)));

    if (lane_id == leader_lane)
    {
      output[word_index(item_index)] = ballot;
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

template <typename InputIterator, typename Predicate>
void launch_valid_if_kernel(
  bitmask_type* output,
  InputIterator begin,
  size_type size,
  Predicate predicate,
  size_type* valid_count,
  cudaStream_t stream)
{
  if (size <= 0)
  {
    return;
  }

  constexpr size_type block_size = 256;
  const size_type num_blocks     = (size + block_size - 1) / block_size;

  valid_if_kernel<block_size><<<num_blocks, block_size, 0, stream>>>(output, begin, size, predicate, valid_count);
}
} // namespace

void cudf_valid_if_custom_kernel(nvbench::state& state)
try
{
  constexpr size_type num_items = 64;
  constexpr size_type num_words = num_items / mask_word_bits;

  thrust::device_vector<int> input(num_items);
  thrust::sequence(input.begin(), input.end(), 0);

  thrust::device_vector<bitmask_type> mask_words(num_words, thrust::no_init);
  thrust::device_vector<size_type> valid_count(1, thrust::no_init);

  auto* d_input       = thrust::raw_pointer_cast(input.data());
  auto* d_mask_words  = thrust::raw_pointer_cast(mask_words.data());
  auto* d_valid_count = thrust::raw_pointer_cast(valid_count.data());

  state.add_element_count(num_items);
  state.add_global_memory_reads<int>(num_items, "Input");
  state.add_global_memory_writes<bitmask_type>(num_words, "MaskWords");
  state.add_global_memory_writes<size_type>(1, "ValidCount");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const cudaStream_t stream = launch.get_stream().get_stream();

    cudaMemsetAsync(d_valid_count, 0, sizeof(size_type), stream);
    launch_valid_if_kernel(
      d_mask_words,
      d_input,
      num_items,
      [] __device__(int value) {
        return (value & 1) == 0;
      },
      d_valid_count,
      stream);
  });

  check_valid_if_correctness(mask_words, valid_count);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(cudf_valid_if_custom_kernel).set_name("cudf_valid_if_custom_kernel");
