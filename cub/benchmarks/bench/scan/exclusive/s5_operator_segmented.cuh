#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/iterator>

#include <nvbench_helper.cuh>

namespace s5_operator_segmented
{
template <typename T>
using s5_state = cuda::std::tuple<T, T, int>;

template <typename T>
struct S5Operator2D
{
  __host__ __device__ s5_state<T> operator()(s5_state<T> x, s5_state<T> y) const
  {
    auto& [x_A, x_Bu, x_flag] = x;
    auto& [y_A, y_Bu, y_flag] = y;

    return {y_flag ? y_A : y_A * x_A, y_flag ? y_Bu : y_A * x_Bu + y_Bu, x_flag | y_flag};
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
struct FlagTransform
{
  int nrows;

  __host__ __device__ int operator()(int k) const
  {
    return k % nrows == 0 ? 1 : 0;
  }
};

template <typename T, int StateDim>
auto setup_scan(
  thrust::device_vector<T>& d_A_in, thrust::device_vector<T>& d_Bu_in, int timesteps, size_t& temp_storage_bytes)
{
  thrust::device_vector<T> d_A_out(timesteps * StateDim, thrust::no_init);
  thrust::device_vector<T> d_Bu_out(timesteps * StateDim, thrust::no_init);

  const int nrows = timesteps;
  const int ncols = StateDim;

  auto col_major_iter =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), ColMajorTransform<T>{nrows, ncols});

  auto A_in_iter  = thrust::make_permutation_iterator(thrust::raw_pointer_cast(d_A_in.data()), col_major_iter);
  auto Bu_in_iter = thrust::make_permutation_iterator(thrust::raw_pointer_cast(d_Bu_in.data()), col_major_iter);
  auto flag_iter  = thrust::make_transform_iterator(thrust::make_counting_iterator(0), FlagTransform<T>{nrows});

  auto input_iter = thrust::make_zip_iterator(A_in_iter, Bu_in_iter, flag_iter);

  auto A_output_iter  = thrust::make_permutation_iterator(d_A_out.begin(), col_major_iter);
  auto Bu_output_iter = thrust::make_permutation_iterator(d_Bu_out.begin(), col_major_iter);

  auto output_iter = thrust::make_zip_iterator(A_output_iter, Bu_output_iter, thrust::make_discard_iterator());

  cub::DeviceScan::InclusiveScan(
    nullptr, temp_storage_bytes, input_iter, output_iter, S5Operator2D<T>(), timesteps * StateDim);

  return input_iter;
}

template <typename T, int StateDim, typename InputIterator, typename StreamT>
void run_scan(void* d_temp_storage,
              size_t temp_storage_bytes,
              InputIterator input_iter,
              thrust::device_vector<T>& d_A_out,
              thrust::device_vector<T>& d_Bu_out,
              int timesteps,
              StreamT stream)
{
  const int nrows = timesteps;
  const int ncols = StateDim;

  auto col_major_iter =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0), ColMajorTransform<T>{nrows, ncols});

  auto A_output_iter  = thrust::make_permutation_iterator(d_A_out.begin(), col_major_iter);
  auto Bu_output_iter = thrust::make_permutation_iterator(d_Bu_out.begin(), col_major_iter);

  auto output_iter = thrust::make_zip_iterator(A_output_iter, Bu_output_iter, thrust::make_discard_iterator());

  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator2D<T>(), timesteps * StateDim, stream);
}
} // namespace s5_operator_segmented
