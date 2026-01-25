// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_partition.cuh>

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

// Predicate for first partition: select if differ result is 1
struct select_first_part_op
{
  __host__ __device__ bool operator()(const cuda::std::tuple<int, int>& t) const
  {
    return cuda::std::get<1>(t) == 1;
  }
};

// Predicate for second partition: dummy (never selects anything)
struct select_second_part_op
{
  template <typename T>
  __host__ __device__ bool operator()(const T&) const
  {
    return false;
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

// Functor for differ: returns 1 if elements are not equal, 0 otherwise
struct neq_op
{
  template <typename Tuple>
  __host__ __device__ int operator()(const Tuple& t) const
  {
    return cuda::std::get<0>(t) != cuda::std::get<1>(t) ? 1 : 0;
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

template <typename OffsetT>
void differ_where_array(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using index_t = int;
  using value_t = int;
  using tuple_t = cuda::std::tuple<index_t, value_t>;

  const auto n = static_cast<OffsetT>(state.get_int64("Elements"));

  // Output iterator - only first_part gets real storage (selected indices)
  // Second part and unselected use discard iterators (dummy, never used)
  thrust::device_vector<index_t> first_part_indices(n);

  auto out_first =
    thrust::make_zip_iterator(cuda::std::make_tuple(first_part_indices.begin(), thrust::make_discard_iterator()));
  // Discard iterators for second part and unselected (not used in select_if pattern)
  auto out_second =
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
  auto out_unselected =
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

  std::vector<int> h_data(static_cast<std::size_t>(n));
  std::srand(42);
  for (int i = 0; i < n; ++i)
  {
    h_data[static_cast<std::size_t>(i)] = std::rand() % 2;
  }
  thrust::device_vector<int> sushi_data(h_data.begin(), h_data.end());

  auto zip_adj     = thrust::make_zip_iterator(cuda::std::make_tuple(sushi_data.begin(), sushi_data.begin() + 1));
  auto differ_iter = thrust::make_transform_iterator(zip_adj, neq_op{});
  auto in_iter     = thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_counting_iterator(1), differ_iter));

  // Number of selected items output (2-element array: [num_first, num_second])
  thrust::device_vector<int> d_num_selected(2);
  int* d_num_selected_out = thrust::raw_pointer_cast(d_num_selected.data());

  // Temporary storage
  std::size_t temp_storage_bytes = 0;
  cub::DevicePartition::If(
    nullptr,
    temp_storage_bytes,
    in_iter,
    out_first,
    out_second,
    out_unselected,
    d_num_selected_out,
    n - 1,
    select_first_part_op{},
    select_second_part_op{});

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<index_t>(n);

  state.exec([&](nvbench::launch& launch) {
    cub::DevicePartition::If(
      d_temp_storage,
      temp_storage_bytes,
      in_iter,
      out_first,
      out_second,
      out_unselected,
      d_num_selected_out,
      n - 1,
      select_first_part_op{},
      select_second_part_op{},
      launch.get_stream());
  });
}

template <typename OffsetT>
void differ_where_lazy(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using index_t = int;
  using value_t = int;
  using tuple_t = cuda::std::tuple<index_t, value_t>;

  const auto n = static_cast<OffsetT>(state.get_int64("Elements"));

  // Output iterator - only first_part gets real storage (selected indices)
  // Second part and unselected use discard iterators (dummy, never used)
  thrust::device_vector<index_t> first_part_indices(n);

  auto out_first =
    thrust::make_zip_iterator(cuda::std::make_tuple(first_part_indices.begin(), thrust::make_discard_iterator()));
  // Discard iterators for second part and unselected (not used in select_if pattern)
  auto out_second =
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
  auto out_unselected =
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

  // Step 1: Create the base iterator for rand(): ZipIterator(CountingIterator, ConstantIterator(2))
  auto base_iter =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), thrust::make_constant_iterator(2)));

  // Step 2: Apply rand_op to generate random values: TransformIterator(base_iter, rand_op)
  auto rand_iter = thrust::make_transform_iterator(base_iter, rand_op<int>{});

  // Step 3: Create PermutationIterators for differ() - accessing values[i] and values[i+1]
  auto zip_adj = thrust::make_zip_iterator(cuda::std::make_tuple(rand_iter, rand_iter + 1));

  // Step 5: TransformIterator applies neq_op to each tuple from zip_adj
  auto differ_iter = thrust::make_transform_iterator(zip_adj, neq_op{});

  // Step 6: ZipIterator combines index (starting at 0) with the differ result
  auto in_iter = thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_counting_iterator(1), differ_iter));

  // Number of selected items output (2-element array: [num_first, num_second])
  thrust::device_vector<int> d_num_selected(2);
  int* d_num_selected_out = thrust::raw_pointer_cast(d_num_selected.data());

  // Temporary storage
  std::size_t temp_storage_bytes = 0;
  cub::DevicePartition::If(
    nullptr,
    temp_storage_bytes,
    in_iter,
    out_first,
    out_second,
    out_unselected,
    d_num_selected_out,
    n - 1,
    select_first_part_op{},
    select_second_part_op{});

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<index_t>(n);

  state.exec([&](nvbench::launch& launch) {
    cub::DevicePartition::If(
      d_temp_storage,
      temp_storage_bytes,
      in_iter,
      out_first,
      out_second,
      out_unselected,
      d_num_selected_out,
      n - 1,
      select_first_part_op{},
      select_second_part_op{},
      launch.get_stream());
  });
}

template <typename OffsetT>
void differ_where_lazy_rand_only(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using index_t = int;
  using value_t = int;
  using tuple_t = cuda::std::tuple<index_t, value_t>;

  const auto n = static_cast<OffsetT>(state.get_int64("Elements"));

  // Output iterator - only first_part gets real storage (selected indices)
  // Second part and unselected use discard iterators (dummy, never used)
  thrust::device_vector<index_t> first_part_indices(n);

  auto out_first =
    thrust::make_zip_iterator(cuda::std::make_tuple(first_part_indices.begin(), thrust::make_discard_iterator()));
  auto out_second =
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));
  auto out_unselected =
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_discard_iterator(), thrust::make_discard_iterator()));

  // Use a cheap mask to control selectivity (~50%)
  auto mask_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), mask_op{});

  // ZipIterator combines index (starting at 1) with the mask result
  auto in_iter = thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_counting_iterator(1), mask_iter));

  // Number of selected items output (2-element array: [num_first, num_second])
  thrust::device_vector<int> d_num_selected(2);
  int* d_num_selected_out = thrust::raw_pointer_cast(d_num_selected.data());

  // Temporary storage
  std::size_t temp_storage_bytes = 0;
  cub::DevicePartition::If(
    nullptr,
    temp_storage_bytes,
    in_iter,
    out_first,
    out_second,
    out_unselected,
    d_num_selected_out,
    n,
    select_first_part_op{},
    select_second_part_op{});

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<index_t>(n);

  state.exec([&](nvbench::launch& launch) {
    cub::DevicePartition::If(
      d_temp_storage,
      temp_storage_bytes,
      in_iter,
      out_first,
      out_second,
      out_unselected,
      d_num_selected_out,
      n,
      select_first_part_op{},
      select_second_part_op{},
      launch.get_stream());
  });
}

template <typename OffsetT>
void differ_where_array_mask_only(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using index_t = int;
  using value_t = int;

  const auto n = static_cast<OffsetT>(state.get_int64("Elements"));

  // Input array (mask values) generated on host to match parrot
  std::vector<int> h_data(static_cast<std::size_t>(n));
  std::srand(42);
  for (int i = 0; i < n; ++i)
  {
    h_data[static_cast<std::size_t>(i)] = std::rand() % 2;
  }
  thrust::device_vector<int> d_in(h_data.begin(), h_data.end());

  // Output iterator - only first_part gets real storage (selected values)
  thrust::device_vector<index_t> out_first(n);
  auto out_second     = thrust::make_discard_iterator();
  auto out_unselected = thrust::make_discard_iterator();

  // Number of selected items output (2-element array: [num_first, num_second])
  thrust::device_vector<int> d_num_selected(2);
  int* d_num_selected_out = thrust::raw_pointer_cast(d_num_selected.data());

  // Temporary storage
  std::size_t temp_storage_bytes = 0;
  cub::DevicePartition::If(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_in.data()),
    out_first.begin(),
    out_second,
    out_unselected,
    d_num_selected_out,
    n,
    select_value_op{},
    select_second_part_op{});

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<index_t>(n);

  state.exec([&](nvbench::launch& launch) {
    cub::DevicePartition::If(
      d_temp_storage,
      temp_storage_bytes,
      thrust::raw_pointer_cast(d_in.data()),
      out_first.begin(),
      out_second,
      out_unselected,
      d_num_selected_out,
      n,
      select_value_op{},
      select_second_part_op{},
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(differ_where_array, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
  .set_name("cub::DevicePartition::If (differ_where_array)")
  .add_int64_axis("Elements", {100000000});

NVBENCH_BENCH_TYPES(differ_where_lazy, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
  .set_name("cub::DevicePartition::If (differ_where_lazy)")
  .add_int64_axis("Elements", {100000000});

NVBENCH_BENCH_TYPES(differ_where_lazy_rand_only, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
  .set_name("cub::DevicePartition::If (differ_where_lazy_rand_only)")
  .add_int64_axis("Elements", {100000000});

NVBENCH_BENCH_TYPES(differ_where_array_mask_only, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
  .set_name("cub::DevicePartition::If (differ_where_array_mask_only)")
  .add_int64_axis("Elements", {100000000});
