// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cstdint>
#include <stdexcept>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

struct block_load_direct_t
{
  static constexpr cub::BlockLoadAlgorithm value = cub::BLOCK_LOAD_DIRECT;
};

struct block_load_striped_t
{
  static constexpr cub::BlockLoadAlgorithm value = cub::BLOCK_LOAD_STRIPED;
};

struct block_load_vectorize_t
{
  static constexpr cub::BlockLoadAlgorithm value = cub::BLOCK_LOAD_VECTORIZE;
};

struct block_load_transpose_t
{
  static constexpr cub::BlockLoadAlgorithm value = cub::BLOCK_LOAD_TRANSPOSE;
};

struct block_load_warp_transpose_t
{
  static constexpr cub::BlockLoadAlgorithm value = cub::BLOCK_LOAD_WARP_TRANSPOSE;
};

struct block_load_warp_transpose_timesliced_t
{
  static constexpr cub::BlockLoadAlgorithm value = cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;
};

#if _CCCL_CTK_BELOW(13, 0)
struct alignas(32) aligned32_t
{
  longlong4 data;
};
#endif // _CCCL_CTK_BELOW(13, 0)

template <int Bytes>
struct transform_load_store_type;

template <>
struct transform_load_store_type<1>
{
  using type = ::cuda::std::int8_t;
};

template <>
struct transform_load_store_type<2>
{
  using type = ::cuda::std::int16_t;
};

template <>
struct transform_load_store_type<4>
{
  using type = ::cuda::std::int32_t;
};

template <>
struct transform_load_store_type<8>
{
  using type = ::cuda::std::int64_t;
};

template <>
struct transform_load_store_type<16>
{
  using type = int4;
};

template <>
struct transform_load_store_type<32>
{
#if _CCCL_CTK_BELOW(13, 0)
  using type = aligned32_t;
#else // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^ / vvv _CCCL_CTK_AT_LEAST(13, 0) vvv
  using type = longlong4_32a;
#endif // _CCCL_CTK_AT_LEAST(13, 0)
};

template <typename T, int ItemsPerThread, ::cuda::std::size_t Alignment>
struct aligned_item_array
{
  alignas(Alignment) char storage[ItemsPerThread * sizeof(T)];

  __device__ __forceinline__ T* data()
  {
    return reinterpret_cast<T*>(storage);
  }

  __device__ __forceinline__ const T* data() const
  {
    return reinterpret_cast<const T*>(storage);
  }

  __device__ __forceinline__ T& operator[](int item)
  {
    return data()[item];
  }

  __device__ __forceinline__ const T& operator[](int item) const
  {
    return data()[item];
  }
};

template <typename T>
__device__ __forceinline__ ::cuda::std::uint32_t checksum_value(T value)
{
  using unsigned_t = ::cuda::std::make_unsigned_t<T>;

  const auto unsigned_value = static_cast<unsigned_t>(value);
  if constexpr (sizeof(unsigned_t) <= sizeof(::cuda::std::uint32_t))
  {
    return static_cast<::cuda::std::uint32_t>(unsigned_value);
  }
  else
  {
    return static_cast<::cuda::std::uint32_t>(unsigned_value)
         ^ static_cast<::cuda::std::uint32_t>(unsigned_value >> 32);
  }
}

__device__ __forceinline__ ::cuda::std::uint32_t mix_checksum(::cuda::std::uint32_t acc, ::cuda::std::uint32_t value)
{
  acc ^= value + 0x9e3779b9u + (acc << 6) + (acc >> 2);
  return (acc << 13) | (acc >> 19);
}

template <typename T, int ItemsPerThread, int VecSize, int BlockThreads, typename ItemArray>
__device__ __forceinline__ void transform_vectorized_load(const T* input, ItemArray& items)
{
  static_assert(ItemsPerThread % VecSize == 0);

  constexpr int load_store_count = ItemsPerThread / VecSize;
  using load_t                   = typename transform_load_store_type<sizeof(T) * VecSize>::type;

  const auto* input_vec = reinterpret_cast<const load_t*>(input) + threadIdx.x;
  auto* item_vec        = reinterpret_cast<load_t*>(items.data());

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int load = 0; load < load_store_count; ++load)
  {
    item_vec[load] = input_vec[load * BlockThreads];
  }
}

template <typename T, int ItemsPerThread, int InputStreams, cub::BlockLoadAlgorithm LoadAlgorithm, int BlockThreads>
__launch_bounds__(BlockThreads) __global__
  void block_load_vectorized_unpack_kernel(const T* input0, const T* input1, const T* input2, const T* input3)
{
  using block_load_t   = cub::BlockLoad<T, BlockThreads, ItemsPerThread, LoadAlgorithm>;
  using block_reduce_t = cub::BlockReduce<::cuda::std::uint32_t, BlockThreads>;

  static_assert(InputStreams == 1 || InputStreams == 4);

  constexpr int tile_items = BlockThreads * ItemsPerThread;

  __shared__ typename block_load_t::TempStorage load_storage[InputStreams];
  __shared__ typename block_reduce_t::TempStorage reduce_storage;

  T items0[ItemsPerThread];
  T items1[ItemsPerThread];
  T items2[ItemsPerThread];
  T items3[ItemsPerThread];
  const auto tile_offset = static_cast<::cuda::std::size_t>(blockIdx.x) * tile_items;

  block_load_t(load_storage[0]).Load(input0 + tile_offset, items0);
  if constexpr (InputStreams == 4)
  {
    block_load_t(load_storage[1]).Load(input1 + tile_offset, items1);
    block_load_t(load_storage[2]).Load(input2 + tile_offset, items2);
    block_load_t(load_storage[3]).Load(input3 + tile_offset, items3);
  }

  auto thread_checksum = static_cast<::cuda::std::uint32_t>(threadIdx.x) ^ 0x811c9dc5u;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    auto value = checksum_value(items0[item]);
    if constexpr (InputStreams == 4)
    {
      const auto value1 = checksum_value(items1[item]);
      const auto value2 = checksum_value(items2[item]);
      const auto value3 = checksum_value(items3[item]);
      value += (value1 * 3u) + (value2 * 5u) + (value3 * 7u);
    }
    thread_checksum = mix_checksum(thread_checksum, value);
  }

  const auto block_checksum = block_reduce_t(reduce_storage).Sum(thread_checksum);
  if (threadIdx.x == 0)
  {
    sink(block_checksum);
  }
}

template <typename T, int ItemsPerThread, int VecSize, int InputStreams, int BlockThreads>
__launch_bounds__(BlockThreads) __global__
  void transform_vectorized_unpack_kernel(const T* input0, const T* input1, const T* input2, const T* input3)
{
  using block_reduce_t = cub::BlockReduce<::cuda::std::uint32_t, BlockThreads>;
  using item_array_t   = aligned_item_array<T, ItemsPerThread, sizeof(T) * VecSize>;

  static_assert(InputStreams == 1 || InputStreams == 4);
  static_assert(ItemsPerThread % VecSize == 0);

  constexpr int tile_items = BlockThreads * ItemsPerThread;

  __shared__ typename block_reduce_t::TempStorage reduce_storage;

  item_array_t items0;
  item_array_t items1;
  item_array_t items2;
  item_array_t items3;
  const auto tile_offset = static_cast<::cuda::std::size_t>(blockIdx.x) * tile_items;

  transform_vectorized_load<T, ItemsPerThread, VecSize, BlockThreads>(input0 + tile_offset, items0);
  if constexpr (InputStreams == 4)
  {
    transform_vectorized_load<T, ItemsPerThread, VecSize, BlockThreads>(input1 + tile_offset, items1);
    transform_vectorized_load<T, ItemsPerThread, VecSize, BlockThreads>(input2 + tile_offset, items2);
    transform_vectorized_load<T, ItemsPerThread, VecSize, BlockThreads>(input3 + tile_offset, items3);
  }

  auto thread_checksum = static_cast<::cuda::std::uint32_t>(threadIdx.x) ^ 0x811c9dc5u;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    auto value = checksum_value(items0[item]);
    if constexpr (InputStreams == 4)
    {
      const auto value1 = checksum_value(items1[item]);
      const auto value2 = checksum_value(items2[item]);
      const auto value3 = checksum_value(items3[item]);
      value += (value1 * 3u) + (value2 * 5u) + (value3 * 7u);
    }
    thread_checksum = mix_checksum(thread_checksum, value);
  }

  const auto block_checksum = block_reduce_t(reduce_storage).Sum(thread_checksum);
  if (threadIdx.x == 0)
  {
    sink(block_checksum);
  }
}

template <typename T, typename ItemsPerThreadT, typename InputStreamsT, typename LoadAlgorithmT>
void block_load_vectorized_unpack(nvbench::state& state,
                                  nvbench::type_list<T, ItemsPerThreadT, InputStreamsT, LoadAlgorithmT>)
try
{
  constexpr int block_threads    = 256;
  constexpr int items_per_thread = ItemsPerThreadT::value;
  constexpr int input_streams    = InputStreamsT::value;
  constexpr auto load_algorithm  = LoadAlgorithmT::value;
  constexpr int tile_items       = block_threads * items_per_thread;

  static_assert(input_streams == 1 || input_streams == 4);

  const auto requested_elements = static_cast<::cuda::std::size_t>(state.get_int64("Elements{io}"));
  const auto elements           = requested_elements - (requested_elements % tile_items);
  const auto num_tiles          = elements / tile_items;

  if (num_tiles == 0)
  {
    state.skip("Elements must cover at least one full BlockLoad tile.");
    return;
  }

  const auto input_elements = elements * input_streams;

  thrust::device_vector<T> input(input_elements);
  thrust::sequence(input.begin(), input.end(), T{1});

  const auto* d_input0 = thrust::raw_pointer_cast(input.data());
  const auto* d_input1 = input_streams == 4 ? d_input0 + elements : d_input0;
  const auto* d_input2 = input_streams == 4 ? d_input1 + elements : d_input0;
  const auto* d_input3 = input_streams == 4 ? d_input2 + elements : d_input0;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements * input_streams);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    block_load_vectorized_unpack_kernel<T, items_per_thread, input_streams, load_algorithm, block_threads>
      <<<static_cast<unsigned int>(num_tiles), block_threads, 0, launch.get_stream()>>>(
        d_input0, d_input1, d_input2, d_input3);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename T, typename ItemsPerThreadT, typename InputStreamsT>
void transform_vectorized_unpack(nvbench::state& state, nvbench::type_list<T, ItemsPerThreadT, InputStreamsT>)
try
{
  constexpr int block_threads    = 256;
  constexpr int items_per_thread = ItemsPerThreadT::value;
  constexpr int input_streams    = InputStreamsT::value;
  constexpr int vec_size         = 4;
  constexpr int tile_items       = block_threads * items_per_thread;

  static_assert(input_streams == 1 || input_streams == 4);
  static_assert(items_per_thread % vec_size == 0);

  const auto requested_elements = static_cast<::cuda::std::size_t>(state.get_int64("Elements{io}"));
  const auto elements           = requested_elements - (requested_elements % tile_items);
  const auto num_tiles          = elements / tile_items;

  if (num_tiles == 0)
  {
    state.skip("Elements must cover at least one full vectorized tile.");
    return;
  }

  const auto input_elements = elements * input_streams;

  thrust::device_vector<T> input(input_elements);
  thrust::sequence(input.begin(), input.end(), T{1});

  const auto* d_input0 = thrust::raw_pointer_cast(input.data());
  const auto* d_input1 = input_streams == 4 ? d_input0 + elements : d_input0;
  const auto* d_input2 = input_streams == 4 ? d_input1 + elements : d_input0;
  const auto* d_input3 = input_streams == 4 ? d_input2 + elements : d_input0;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements * input_streams);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    transform_vectorized_unpack_kernel<T, items_per_thread, vec_size, input_streams, block_threads>
      <<<static_cast<unsigned int>(num_tiles), block_threads, 0, launch.get_stream()>>>(
        d_input0, d_input1, d_input2, d_input3);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using value_types =
  nvbench::type_list<::cuda::std::uint8_t, ::cuda::std::uint16_t, ::cuda::std::uint32_t, ::cuda::std::uint64_t>;

using items_per_thread =
  nvbench::type_list<::cuda::std::integral_constant<int, 2>, ::cuda::std::integral_constant<int, 4>>;

using transform_items_per_thread =
  nvbench::type_list<::cuda::std::integral_constant<int, 4>, ::cuda::std::integral_constant<int, 8>>;

using input_stream_counts =
  nvbench::type_list<::cuda::std::integral_constant<int, 1>, ::cuda::std::integral_constant<int, 4>>;

using load_algorithms =
  nvbench::type_list<block_load_direct_t,
                     block_load_striped_t,
                     block_load_vectorize_t,
                     block_load_transpose_t,
                     block_load_warp_transpose_t,
                     block_load_warp_transpose_timesliced_t>;

NVBENCH_BENCH_TYPES(block_load_vectorized_unpack,
                    NVBENCH_TYPE_AXES(value_types, items_per_thread, input_stream_counts, load_algorithms))
  .set_name("vectorized_unpack")
  .set_type_axes_names({"T{ct}", "ItemsPerThread{ct}", "InputStreams{ct}", "LoadAlgorithm{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(20, 28, 4));

NVBENCH_BENCH_TYPES(transform_vectorized_unpack,
                    NVBENCH_TYPE_AXES(value_types, transform_items_per_thread, input_stream_counts))
  .set_name("transform_vectorized_unpack")
  .set_type_axes_names({"T{ct}", "ItemsPerThread{ct}", "InputStreams{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(20, 28, 4));
