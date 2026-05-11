// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_hierarchical_transform.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
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
    throw std::runtime_error("hierarchical_valid_if_ballot correctness check failed: unexpected number of mask words.");
  }

  for (std::size_t word = 0; word < h_mask_words.size(); ++word)
  {
    if (h_mask_words[word] != expected_word)
    {
      throw std::runtime_error("hierarchical_valid_if_ballot correctness check failed: unexpected bitmask word.");
    }
  }

  if (h_valid_count.size() != 1 || h_valid_count[0] != expected_valid_count)
  {
    throw std::runtime_error("hierarchical_valid_if_ballot correctness check failed: unexpected valid count.");
  }
}
} // namespace

void hierarchical_valid_if_ballot(nvbench::state& state)
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

  state.add_element_count(num_items);
  state.add_global_memory_reads<int>(num_items, "Input");
  state.add_global_memory_writes<std::uint32_t>(num_words, "MaskWords");
  state.add_global_memory_writes<int>(1, "ValidCount");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const cudaStream_t stream = launch.get_stream().get_stream();
    cudaMemsetAsync(d_valid_count, 0, sizeof(int), stream);

    cub::DeviceSegmentedTransform::TransformEpilog(
      d_input,
      cuda::discard_iterator{},
      num_words,
      mask_word_bits,
      [] __device__(int value) -> bool {
        return (value & 1) == 0;
      },
      [d_mask_words, d_valid_count] __device__(auto block_group, const auto& results, const auto& indices) {
        using results_t                = cuda::std::remove_reference_t<decltype(results)>;
        constexpr int items_per_thread = cuda::std::extent_v<results_t>;
        static_assert(mask_word_bits % items_per_thread == 0);

        constexpr int subgroup_size = mask_word_bits / items_per_thread;
        const int lane_rank         = static_cast<int>(threadIdx.x % mask_word_bits);
        const int subgroup_rank     = lane_rank / subgroup_size;
        const int subgroup_lane     = lane_rank % subgroup_size;

        const cuda::std::uint32_t subgroup_mask =
          items_per_thread == 1 ? 0xffffffffu : ((1u << subgroup_size) - 1u) << (subgroup_rank * subgroup_size);

        cuda::std::uint32_t local_mask = 0;
        cuda::std::int64_t word_index  = -1;

        for (int item = 0; item < items_per_thread; ++item)
        {
          const auto index = indices[item];
          if (index >= 0)
          {
            word_index = index / mask_word_bits;
          }

          if (index >= 0 && results[item])
          {
            local_mask |= 1u << (subgroup_lane * items_per_thread + item);
          }
        }

        const cuda::std::uint32_t word = __reduce_or_sync(subgroup_mask, local_mask);
        int warp_valid                 = 0;

        if (subgroup_lane == 0 && word_index >= 0)
        {
          d_mask_words[word_index] = word;
          warp_valid               = __popc(word);
        }

        const int block_valid = cuda::coop::reduce(block_group, warp_valid, cuda::std::plus<>{});

        if (cuda::gpu_thread.is_root_rank(block_group))
        {
          cuda::atomic_ref<int, cuda::thread_scope_device> atomic_valid_count(*d_valid_count);
          atomic_valid_count.fetch_add(block_valid, cuda::memory_order_relaxed);
        }
      },
      stream);
  });

  check_valid_if_correctness(mask_words, valid_count);
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH(hierarchical_valid_if_ballot).set_name("hierarchical_valid_if_ballot");
