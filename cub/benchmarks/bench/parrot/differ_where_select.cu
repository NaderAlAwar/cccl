// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/tuple.h>

#include <cuda/std/tuple>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>
#ifndef __CUDA_ARCH__
#  include <random>
#endif
#include <type_traits>

#include <nvbench_helper.cuh>

/*
 * Benchmark for sushi.differ().where() chain using CUB DeviceSelect::If.
 *
 * This is similar to differ_where.cu but uses select_if instead of three_way_partition.
 *
 * Iterator structure:
 * - in_iter = ZipIterator
 *     ├── CountingIterator(1)  (tracks indices starting at 1)
 *     └── TransformIterator(neq_op)
 *         └── ZipIterator
 *             ├── PermutationIterator(sushi, CountingIterator(0))  → sushi[i]
 *             └── PermutationIterator(sushi, CountingIterator(1))  → sushi[i+1]
 *
 * Predicate selects where mask (differ result) is truthy.
 */

// neq_op: returns 1 if tuple elements differ, 0 otherwise
struct neq_op
{
  template <typename Tuple>
  __host__ __device__ int operator()(const Tuple& t) const
  {
    return (cuda::std::get<0>(t) != cuda::std::get<1>(t)) ? 1 : 0;
  }
};

// rand_op: generates pseudo-random value from (index, constant_val) tuple
// Returns value in range [0, constant_val)
template <typename T>
struct rand_op
{
  unsigned int extra_entropy;

  __host__ __device__ rand_op()
  {
#ifndef __CUDA_ARCH__
    std::random_device rd;
    extra_entropy = rd() ^ (static_cast<unsigned int>(clock()) * 2654435761U)
                  ^ static_cast<unsigned int>(reinterpret_cast<uintptr_t>(this) & 0xFFFFFFFF);
#else
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extra_entropy    = 0x9e3779b9 + tid + clock();
#endif
  }

  __host__ __device__ auto operator()(const thrust::tuple<int, T>& t) const -> T
  {
    int idx = thrust::get<0>(t);
    T val   = thrust::get<1>(t);

    auto h1 = static_cast<unsigned int>(idx ^ extra_entropy);
    h1      = ((h1 >> 16) ^ h1) * 0x45d9f3b;
    h1      = ((h1 >> 16) ^ h1) * 0x45d9f3b;
    h1      = (h1 >> 16) ^ h1;

    thrust::default_random_engine rng(static_cast<thrust::default_random_engine::result_type>(h1));
    thrust::uniform_real_distribution<float> dist(0.0F, 1.0F);
    float rand_val = dist(rng);

    if (std::is_integral<T>::value)
    {
      return static_cast<T>(rand_val * val);
    }
    return static_cast<T>(rand_val * val);
  }
};

struct mask_op
{
  __host__ __device__ int operator()(int i) const
  {
    return i & 1;
  }
};

// Predicate for array-based mask: select if value is non-zero
struct select_value_op
{
  __host__ __device__ bool operator()(int v) const
  {
    return v != 0;
  }
};

template <typename T, typename OffsetT>
void differ_where_select_array(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  std::vector<T> h_data(static_cast<std::size_t>(n));
  std::srand(42);
  for (int i = 0; i < n; ++i)
  {
    h_data[static_cast<std::size_t>(i)] = static_cast<T>(std::rand() % 2);
  }
  thrust::device_vector<T> sushi_data(h_data.begin(), h_data.end());

  auto zip_adj = thrust::make_zip_iterator(cuda::std::make_tuple(sushi_data.begin(), sushi_data.begin() + 1));

  auto differ_iter = thrust::make_transform_iterator(zip_adj, neq_op{});

  auto in_iter = thrust::make_counting_iterator(1);

  // n-1 elements (differ reduces length by 1)
  OffsetT differ_len = static_cast<OffsetT>(n - 1);

  // Output: indices where mask is non-zero
  thrust::device_vector<int> d_out(differ_len);
  thrust::device_vector<OffsetT> d_num_selected(1);

  // Add metrics
  state.add_element_count(differ_len);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<int>(differ_len / 2); // Approximate output

  // Query temp storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(
    nullptr,
    temp_storage_bytes,
    in_iter,
    differ_iter,
    d_out.begin(),
    thrust::raw_pointer_cast(d_num_selected.data()),
    differ_len);

  // Allocate temp storage
  thrust::device_vector<nvbench::uint8_t> d_temp_storage(temp_storage_bytes);

  // Run benchmark
  state.exec([&](nvbench::launch& launch) {
    cub::DeviceSelect::Flagged(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      in_iter,
      differ_iter,
      d_out.begin(),
      thrust::raw_pointer_cast(d_num_selected.data()),
      differ_len,
      launch.get_stream());
  });
}

template <typename T, typename OffsetT>
void differ_where_select_lazy(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  // Step 1: Create the base iterator for rand(): ZipIterator(CountingIterator, ConstantIterator(2))
  auto base_iter =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), thrust::make_constant_iterator(2)));

  // Step 2: Apply rand_op to generate random values: TransformIterator(base_iter, rand_op)
  auto rand_iter = thrust::make_transform_iterator(base_iter, rand_op<int>{});

  // Step 3: Create PermutationIterators for differ() - accessing values[i] and values[i+1]
  auto zip_adj = thrust::make_zip_iterator(cuda::std::make_tuple(rand_iter, rand_iter + 1));

  // Step 5: TransformIterator applies neq_op to each tuple from zip_adj
  auto differ_iter = thrust::make_transform_iterator(zip_adj, neq_op{});

  // Step 6: ZipIterator combines index (starting at 1) with the differ result
  auto in_iter = thrust::make_counting_iterator(1);

  // n-1 elements (differ reduces length by 1)
  OffsetT differ_len = static_cast<OffsetT>(n - 1);

  // Output: indices where mask is non-zero
  thrust::device_vector<int> d_out(differ_len);
  thrust::device_vector<OffsetT> d_num_selected(1);

  // Add metrics
  state.add_element_count(differ_len);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<int>(differ_len / 2); // Approximate output

  // Query temp storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(
    nullptr,
    temp_storage_bytes,
    in_iter,
    differ_iter,
    d_out.begin(),
    thrust::raw_pointer_cast(d_num_selected.data()),
    differ_len);

  // Allocate temp storage
  thrust::device_vector<nvbench::uint8_t> d_temp_storage(temp_storage_bytes);

  // Run benchmark
  state.exec([&](nvbench::launch& launch) {
    cub::DeviceSelect::Flagged(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      in_iter,
      differ_iter,
      d_out.begin(),
      thrust::raw_pointer_cast(d_num_selected.data()),
      differ_len,
      launch.get_stream());
  });
}

template <typename T, typename OffsetT>
void differ_where_select_lazy_rand_only(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  // Use a cheap mask to control selectivity (~50%)
  auto mask_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), mask_op{});

  // Step 3: Use 1-indexed range as input (matches parrot::range)
  auto in_iter = thrust::make_counting_iterator(1);

  // Output: indices where mask is non-zero
  thrust::device_vector<int> d_out(n);
  thrust::device_vector<OffsetT> d_num_selected(1);

  // Add metrics
  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<int>(n / 2); // Approximate output

  // Query temp storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(
    nullptr, temp_storage_bytes, in_iter, mask_iter, d_out.begin(), thrust::raw_pointer_cast(d_num_selected.data()), n);

  // Allocate temp storage
  thrust::device_vector<nvbench::uint8_t> d_temp_storage(temp_storage_bytes);

  // Run benchmark
  state.exec([&](nvbench::launch& launch) {
    cub::DeviceSelect::Flagged(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      in_iter,
      mask_iter,
      d_out.begin(),
      thrust::raw_pointer_cast(d_num_selected.data()),
      n,
      launch.get_stream());
  });
}

template <typename T, typename OffsetT>
void differ_where_select_array_mask_only(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  std::vector<T> h_data(static_cast<std::size_t>(n));
  std::srand(42);
  for (int i = 0; i < n; ++i)
  {
    h_data[static_cast<std::size_t>(i)] = static_cast<T>(std::rand() % 2);
  }
  thrust::device_vector<T> d_in(h_data.begin(), h_data.end());

  // Output: selected values
  thrust::device_vector<T> d_out(n);
  thrust::device_vector<OffsetT> d_num_selected(1);

  // Add metrics
  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n / 2); // Approximate output

  // Query temp storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_in.data()),
    d_out.begin(),
    thrust::raw_pointer_cast(d_num_selected.data()),
    n,
    select_value_op{});

  // Allocate temp storage
  thrust::device_vector<nvbench::uint8_t> d_temp_storage(temp_storage_bytes);

  // Run benchmark
  state.exec([&](nvbench::launch& launch) {
    cub::DeviceSelect::If(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_in.data()),
      d_out.begin(),
      thrust::raw_pointer_cast(d_num_selected.data()),
      n,
      select_value_op{},
      launch.get_stream());
  });
}

using my_offset_types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(differ_where_select_array, NVBENCH_TYPE_AXES(nvbench::type_list<int>, my_offset_types))
  .set_name("differ_where_select_array")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000}); // 100M

NVBENCH_BENCH_TYPES(differ_where_select_lazy, NVBENCH_TYPE_AXES(nvbench::type_list<int>, my_offset_types))
  .set_name("differ_where_select_lazy")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000}); // 100M

NVBENCH_BENCH_TYPES(differ_where_select_lazy_rand_only, NVBENCH_TYPE_AXES(nvbench::type_list<int>, my_offset_types))
  .set_name("differ_where_select_lazy_rand_only")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000}); // 100M

NVBENCH_BENCH_TYPES(differ_where_select_array_mask_only, NVBENCH_TYPE_AXES(nvbench::type_list<int>, my_offset_types))
  .set_name("differ_where_select_array_mask_only")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000}); // 100M
