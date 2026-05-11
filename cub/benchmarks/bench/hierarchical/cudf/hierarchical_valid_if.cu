// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_hierarchical_segmented_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/utility>

#include <stdexcept>

#include <nvbench_helper.cuh>

namespace
{
constexpr int mask_word_bits = 32;

void check_valid_if_correctness(const thrust::device_vector<std::uint32_t>& mask_words,
                                const thrust::device_vector<int>& valid_count)
{
  const thrust::host_vector<std::uint32_t> h_mask_words = mask_words;
  const thrust::host_vector<int> h_valid_count          = valid_count;

  constexpr std::uint32_t expected_word = 0x55555555u;
  constexpr int expected_valid_count    = 32;

  if (h_mask_words.size() != 2)
  {
    throw std::runtime_error("hierarchical_valid_if correctness check failed: unexpected number of mask words.");
  }

  for (std::size_t word = 0; word < h_mask_words.size(); ++word)
  {
    if (h_mask_words[word] != expected_word)
    {
      throw std::runtime_error("hierarchical_valid_if correctness check failed: unexpected bitmask word.");
    }
  }

  if (h_valid_count.size() != 1 || h_valid_count[0] != expected_valid_count)
  {
    throw std::runtime_error("hierarchical_valid_if correctness check failed: unexpected valid count.");
  }
}
} // namespace

void hierarchical_valid_if(nvbench::state& state)
try
{
  constexpr int num_items = 64;
  constexpr int num_words = (num_items + mask_word_bits - 1) / mask_word_bits;

  thrust::device_vector<int> input(num_items);
  thrust::sequence(input.begin(), input.end(), 0);

  thrust::device_vector<std::uint32_t> mask_words(num_words, thrust::no_init);
  thrust::device_vector<int> valid_count(1, thrust::no_init);

  auto* d_input       = thrust::raw_pointer_cast(input.data());
  auto* d_mask_words  = thrust::raw_pointer_cast(mask_words.data());
  auto* d_valid_count = thrust::raw_pointer_cast(valid_count.data());

  auto packed_predicate_bits =
    cuda::make_transform_iterator(cuda::counting_iterator<int>{0}, [d_input] __device__(int index) -> std::uint32_t {
      return (d_input[index] & 1) == 0 ? (std::uint32_t{1} << (index % mask_word_bits)) : 0u;
    });

  state.add_element_count(num_items);
  state.add_global_memory_reads<int>(num_items, "Input");
  state.add_global_memory_writes<std::uint32_t>(num_words, "MaskWords");
  state.add_global_memory_writes<int>(1, "ValidCount");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const cudaStream_t stream = launch.get_stream().get_stream();
    cudaMemsetAsync(d_valid_count, 0, sizeof(int), stream);

    cub::DeviceHierarchicalSegmentedReduce::Reduce(
      packed_predicate_bits,
      d_mask_words,
      num_words,
      num_items,
      mask_word_bits,
      cuda::std::bit_or<>{},
      std::uint32_t{0},
      [d_valid_count] __device__(auto group, std::uint32_t thread_partial) -> std::uint32_t {
        const std::uint32_t word = cuda::coop::reduce(group, thread_partial, cuda::std::bit_or<>{});

        if (cuda::gpu_thread.is_root_rank(group))
        {
          cuda::atomic_ref<int, cuda::thread_scope_device> atomic_valid_count(*d_valid_count);
          atomic_valid_count.fetch_add(__popc(word), cuda::memory_order_relaxed);
          return word;
        }

        return 0u;
      },
      stream);
  });

  check_valid_if_correctness(mask_words, valid_count);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(hierarchical_valid_if).set_name("hierarchical_valid_if");
