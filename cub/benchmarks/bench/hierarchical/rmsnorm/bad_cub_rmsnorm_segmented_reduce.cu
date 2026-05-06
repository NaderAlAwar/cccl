// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/transform.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

#include "rmsnorm_check.cuh"

constexpr float rms_norm_eps = 1e-5f;

template <typename T>
struct convert_op
{
  __device__ T operator()(float x) const
  {
    return static_cast<T>(x);
  }
};

template <typename T>
struct square_op
{
  __device__ float operator()(T x) const
  {
    const float value = static_cast<float>(x);
    return value * value;
  }
};

struct reciprocal_rms_op
{
  int hidden_size;
  float eps;

  __device__ float operator()(float sum_of_squares) const
  {
    return rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + eps);
  }
};

template <typename T>
struct segment_write_context
{
  const T* input{};
  T* output{};
  const T* weight{};
  int hidden_size{};
};

template <typename T>
class segment_write_proxy
{
public:
  __host__ __device__ segment_write_proxy(int segment_id, segment_write_context<T> context)
      : segment_id_(segment_id)
      , context_(context)
  {}

  __host__ __device__ segment_write_proxy(const segment_write_proxy&)            = default;
  __host__ __device__ segment_write_proxy& operator=(const segment_write_proxy&) = default;

  __host__ __device__ segment_write_proxy& operator=(float rms_rcp)
  {
    write_segment(rms_rcp);
    return *this;
  }

  __host__ __device__ const segment_write_proxy& operator=(float rms_rcp) const
  {
    write_segment(rms_rcp);
    return *this;
  }

private:
  __host__ __device__ void write_segment(float rms_rcp) const
  {
    const int segment_begin = segment_id_ * context_.hidden_size;

    for (int col = 0; col < context_.hidden_size; ++col)
    {
      const float scale                    = static_cast<float>(context_.weight[col]) * rms_rcp;
      const float value                    = static_cast<float>(context_.input[segment_begin + col]);
      context_.output[segment_begin + col] = static_cast<T>(value * scale);
    }
  }

  int segment_id_;
  segment_write_context<T> context_;
};

template <typename T>
class segment_write_iterator
{
public:
  using iterator_concept  = cuda::std::random_access_iterator_tag;
  using iterator_category = cuda::std::output_iterator_tag;
  using difference_type   = cuda::std::iter_difference_t<cuda::counting_iterator<int>>;
  using value_type        = void;
  using pointer           = void;
  using reference         = segment_write_proxy<T>;

  __host__ __device__ segment_write_iterator() = default;

  __host__ __device__ explicit segment_write_iterator(segment_write_context<T> context, int segment_id = 0)
      : segment_ids_(segment_id)
      , context_(context)
  {}

  __host__ __device__ reference operator*() const
  {
    return reference{*segment_ids_, context_};
  }

  __host__ __device__ reference operator[](difference_type n) const
  {
    return reference{segment_ids_[n], context_};
  }

  __host__ __device__ segment_write_iterator& operator++()
  {
    ++segment_ids_;
    return *this;
  }

  __host__ __device__ segment_write_iterator operator++(int)
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  __host__ __device__ segment_write_iterator& operator--()
  {
    --segment_ids_;
    return *this;
  }

  __host__ __device__ segment_write_iterator operator--(int)
  {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  __host__ __device__ segment_write_iterator& operator+=(difference_type n)
  {
    segment_ids_ += n;
    return *this;
  }

  __host__ __device__ segment_write_iterator& operator-=(difference_type n)
  {
    segment_ids_ -= n;
    return *this;
  }

  friend __host__ __device__ segment_write_iterator operator+(segment_write_iterator it, difference_type n)
  {
    it += n;
    return it;
  }

  friend __host__ __device__ segment_write_iterator operator+(difference_type n, segment_write_iterator it)
  {
    it += n;
    return it;
  }

  friend __host__ __device__ segment_write_iterator operator-(segment_write_iterator it, difference_type n)
  {
    it -= n;
    return it;
  }

  friend __host__ __device__ difference_type operator-(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ - rhs.segment_ids_;
  }

  friend __host__ __device__ bool operator==(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ == rhs.segment_ids_;
  }

  friend __host__ __device__ bool operator!=(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ != rhs.segment_ids_;
  }

  friend __host__ __device__ bool operator<(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ < rhs.segment_ids_;
  }

  friend __host__ __device__ bool operator>(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ > rhs.segment_ids_;
  }

  friend __host__ __device__ bool operator<=(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ <= rhs.segment_ids_;
  }

  friend __host__ __device__ bool operator>=(segment_write_iterator lhs, segment_write_iterator rhs)
  {
    return lhs.segment_ids_ >= rhs.segment_ids_;
  }

private:
  cuda::counting_iterator<int> segment_ids_{};
  segment_write_context<T> context_{};
};

template <typename T>
thrust::device_vector<T> make_bounded_vector(std::size_t elements, bool zero_data)
{
  if (zero_data)
  {
    return thrust::device_vector<T>(elements, T{});
  }

  thrust::device_vector<float> source = generate(elements, bit_entropy::_1_000, -1.0f, 1.0f);

  if constexpr (cuda::std::is_same_v<T, float>)
  {
    return source;
  }
  else
  {
    thrust::device_vector<T> destination(elements, thrust::no_init);
    thrust::transform(source.begin(), source.end(), destination.begin(), convert_op<T>{});
    return destination;
  }
}

template <typename T>
void bad_rmsnorm_segmented_reduce_cub(nvbench::state& state, nvbench::type_list<T>)
try
{
  const int batch_size  = static_cast<int>(state.get_int64("BatchSize"));
  const int hidden_size = static_cast<int>(state.get_int64("HiddenSize"));
  const bool zero_data  = state.get_int64("ZeroData") != 0;

  const auto elements = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(hidden_size);

  thrust::device_vector<T> input = make_bounded_vector<T>(elements, zero_data);
  thrust::device_vector<T> output(elements, thrust::no_init);
  thrust::device_vector<T> weight = make_bounded_vector<T>(static_cast<std::size_t>(hidden_size), zero_data);

  auto* d_input  = thrust::raw_pointer_cast(input.data());
  auto* d_output = thrust::raw_pointer_cast(output.data());
  auto* d_weight = thrust::raw_pointer_cast(weight.data());

  auto squared_input  = cuda::make_transform_iterator(d_input, square_op<T>{});
  auto segment_output = cuda::make_transform_output_iterator(
    segment_write_iterator<T>{segment_write_context<T>{d_input, d_output, d_weight, hidden_size}},
    reciprocal_rms_op{hidden_size, rms_norm_eps});

  std::size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, squared_input, segment_output, batch_size, hidden_size);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  auto* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Input");
  state.add_global_memory_reads<T>(hidden_size, "Weight");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes, squared_input, segment_output, batch_size, hidden_size, launch.get_stream());
  });

#if 1
  rmsnorm_check::check_correctness(batch_size, hidden_size, rms_norm_eps, input, weight, output);
#endif
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types =
  nvbench::type_list<float
#  if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
                     ,
                     __half
#  endif
#  if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
                     ,
                     __nv_bfloat16
#  endif
                     >;
#endif

NVBENCH_BENCH_TYPES(bad_rmsnorm_segmented_reduce_cub, NVBENCH_TYPE_AXES(value_types))
  .set_name("cub_rmsnorm")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BatchSize", {64, 8192, 20000, 75000, 150000, 299000})
  .add_int64_axis("ZeroData", {0, 1})
  .add_int64_axis("HiddenSize", {2880, 7168});
