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

#include <cuda/std/tuple>

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
struct rand_op
{
  unsigned int _entropy = 0;

  template <typename Tuple>
  __host__ __device__ int operator()(const Tuple& t) const
  {
    int idx = cuda::std::get<0>(t);
    int val = cuda::std::get<1>(t);

    // Hash (matches parrot's rand_op)
    unsigned int h = static_cast<unsigned int>(idx) ^ _entropy;
    h              = ((h >> 16) ^ h) * 0x45d9f3b;
    h              = ((h >> 16) ^ h) * 0x45d9f3b;
    h              = (h >> 16) ^ h;

    // Convert to float in [0, 1), multiply by val
    float rand_val = static_cast<float>(h & 0x7FFFFF) / static_cast<float>(0x800000);
    return static_cast<int>(rand_val * static_cast<float>(val));
  }
};

template <typename OffsetT>
void differ_where(nvbench::state& state, nvbench::type_list<OffsetT>)
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

  // ============================================================================
  // OPTION A: Array-based approach (uncomment to use)
  // ============================================================================
  // thrust::device_vector<int> sushi_data = generate(n, bit_entropy::_1_000, int{0}, int{1});

  // auto perm_a = thrust::make_permutation_iterator(
  //     sushi_data.begin(),
  //     thrust::make_counting_iterator(0)
  // );
  // auto perm_b = thrust::make_permutation_iterator(
  //     sushi_data.begin(),
  //     thrust::make_counting_iterator(1)
  // );
  // auto zip_adj = thrust::make_zip_iterator(cuda::std::make_tuple(perm_a, perm_b));
  // auto differ_iter = thrust::make_transform_iterator(zip_adj, neq_op{});
  // auto in_iter = thrust::make_zip_iterator(cuda::std::make_tuple(
  //     thrust::make_counting_iterator(0),
  //     differ_iter
  // ));
  // ============================================================================
  // END OPTION A
  // ============================================================================

  // ============================================================================
  // OPTION B: Parrot-style lazy iterator approach (currently active)
  // Matches: constant(2, n).rand().differ().where()
  // ============================================================================
  // Step 1: Create the base iterator for rand(): ZipIterator(CountingIterator, ConstantIterator(2))
  auto base_iter = thrust::make_zip_iterator(
    cuda::std::make_tuple(thrust::make_counting_iterator(0), thrust::make_constant_iterator(2)));

  // Step 2: Apply rand_op to generate random values: TransformIterator(base_iter, rand_op)
  auto rand_iter = thrust::make_transform_iterator(base_iter, rand_op{});

  // Step 3: Create PermutationIterators for differ() - accessing values[i] and values[i+1]
  auto perm_a = thrust::make_permutation_iterator(rand_iter, thrust::make_counting_iterator(0));
  auto perm_b = thrust::make_permutation_iterator(rand_iter, thrust::make_counting_iterator(1));

  // Step 4: Zip the two permutation iterators for adjacent comparison
  auto zip_adj = thrust::make_zip_iterator(cuda::std::make_tuple(perm_a, perm_b));

  // Step 5: TransformIterator applies neq_op to each tuple from zip_adj
  auto differ_iter = thrust::make_transform_iterator(zip_adj, neq_op{});

  // Step 6: ZipIterator combines index (starting at 0) with the differ result
  auto in_iter = thrust::make_zip_iterator(cuda::std::make_tuple(thrust::make_counting_iterator(0), differ_iter));
  // ============================================================================
  // END OPTION B
  // ============================================================================

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

NVBENCH_BENCH_TYPES(differ_where, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
  .set_name("cub::DevicePartition::If (differ_where)")
  .add_int64_axis("Elements", {100000000});
