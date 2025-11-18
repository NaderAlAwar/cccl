#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

namespace s5_operator_segmented_actual
{
template <typename T>
struct S5Operator2D
{
  __host__ __device__ cuda::std::tuple<T, T> operator()(cuda::std::tuple<T, T> x, cuda::std::tuple<T, T> y) const
  {
    const auto& [x_A, x_Bu] = x;
    const auto& [y_A, y_Bu] = y;

    return {y_A * x_A, y_A * x_Bu + y_Bu};
  }
};

// Helper struct to hold the lambda functors outside the function
template <typename T>
struct ColMajorTransform
{
  int nrows;
  int ncols;

  __host__ __device__ int operator()(int k) const
  {
    int row = k % nrows;
    int col = k / nrows;
    return row * ncols + col;
  }
};

template <typename T>
thrust::device_vector<T> generate_data(std::size_t n)
{
  if constexpr (cuda::std::is_same_v<T, __half>)
  {
    // Generate as float, then convert to __half
    thrust::device_vector<float> temp = generate(n);
    thrust::device_vector<__half> result(n);
    thrust::transform(temp.begin(), temp.end(), result.begin(), [] __device__(float f) {
      return __float2half(f);
    });
    return result;
  }
  else
  {
    return generate(n);
  }
}

template <typename T, int StateDim>
auto setup_scan(
  thrust::device_vector<T>& d_A_in, thrust::device_vector<T>& d_Bu_in, int timesteps, size_t& temp_storage_bytes)
{
  const int nrows = timesteps;
  const int ncols = StateDim;

  auto col_major_iter =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), ColMajorTransform<T>{nrows, ncols});

  auto A_in_iter  = thrust::make_permutation_iterator(thrust::raw_pointer_cast(d_A_in.data()), col_major_iter);
  auto Bu_in_iter = thrust::make_permutation_iterator(thrust::raw_pointer_cast(d_Bu_in.data()), col_major_iter);

  auto input_iter = thrust::make_zip_iterator(A_in_iter, Bu_in_iter);

  thrust::device_vector<int> d_A_out(timesteps * StateDim);
  thrust::device_vector<int> d_Bu_out(timesteps * StateDim);

  auto A_output_iter  = thrust::make_permutation_iterator(d_A_out.begin(), col_major_iter);
  auto Bu_output_iter = thrust::make_permutation_iterator(d_Bu_out.begin(), col_major_iter);

  auto output_iter = thrust::make_zip_iterator(A_output_iter, Bu_output_iter);

  thrust::device_vector<int> begin_offsets(StateDim, thrust::no_init);
  thrust::device_vector<int> end_offsets(StateDim, thrust::no_init);

  thrust::sequence(begin_offsets.begin(), begin_offsets.end(), 0, nrows);
  thrust::sequence(end_offsets.begin(), end_offsets.end(), nrows, nrows);

  cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    nullptr,
    temp_storage_bytes,
    input_iter,
    output_iter,
    thrust::raw_pointer_cast(begin_offsets.data()),
    thrust::raw_pointer_cast(end_offsets.data()),
    StateDim,
    S5Operator2D<T>());

  return std::make_tuple(input_iter, begin_offsets, end_offsets);
}

template <typename T, int StateDim, typename InputIterator, typename StreamT>
void run_scan(
  void* d_temp_storage,
  size_t temp_storage_bytes,
  InputIterator input_iter,
  thrust::device_vector<T>& d_A_out,
  thrust::device_vector<T>& d_Bu_out,
  thrust::device_vector<int>& begin_offsets,
  thrust::device_vector<int>& end_offsets,
  int timesteps,
  StreamT stream)
{
  const int nrows = timesteps;
  const int ncols = StateDim;

  auto col_major_iter =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), ColMajorTransform<T>{nrows, ncols});

  auto A_output_iter  = thrust::make_permutation_iterator(d_A_out.begin(), col_major_iter);
  auto Bu_output_iter = thrust::make_permutation_iterator(d_Bu_out.begin(), col_major_iter);

  auto output_iter = thrust::make_zip_iterator(A_output_iter, Bu_output_iter);

  cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    d_temp_storage,
    temp_storage_bytes,
    input_iter,
    output_iter,
    thrust::raw_pointer_cast(begin_offsets.data()),
    thrust::raw_pointer_cast(end_offsets.data()),
    StateDim,
    S5Operator2D<T>(),
    stream);
}
} // namespace s5_operator_segmented_actual
