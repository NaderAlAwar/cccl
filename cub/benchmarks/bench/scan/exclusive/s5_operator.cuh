// CUB S5 Associative Scan Benchmark
// Demonstrates 1D (scalar) and 2D (vector/row-wise) associative scans

#include <cuda_fp16.h>

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>

#include <nvbench_helper.cuh>

// S5 operator working on thrust::tuple
template <typename T>
struct S5Operator
{
  using Tuple = thrust::tuple<T, T>;

  __host__ __device__ Tuple operator()(const Tuple& x, const Tuple& y) const
  {
    // thrust::get<0> = A, thrust::get<1> = Bu
    // x: (A_i, Bu_i), y: (A_j, Bu_j)
    return thrust::make_tuple(thrust::get<0>(y) * thrust::get<0>(x), // A_j * A_i
                              thrust::get<0>(y) * thrust::get<1>(x) + thrust::get<1>(y) // A_j * Bu_i + Bu_j
    );
  }
};

template <typename T, int HIDDEN_DIM>
using RowPair = cuda::std::pair<cuda::std::array<T, HIDDEN_DIM>, cuda::std::array<T, HIDDEN_DIM>>;

// S5 operator for 2D tensors with elementwise operations
template <typename T, int HIDDEN_DIM>
struct S5Operator2D
{
  __host__ __device__ RowPair<T, HIDDEN_DIM>
  operator()(const RowPair<T, HIDDEN_DIM>& x, const RowPair<T, HIDDEN_DIM>& y) const
  {
    auto& [x_A, x_Bu] = x;
    auto& [y_A, y_Bu] = y;

    RowPair<T, HIDDEN_DIM> result;
    auto& [A, Bu] = result;

// Elementwise operations:
// result.A[i] = y.A[i] * x.A[i]
// result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i]
#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      A[i]  = y_A[i] * x_A[i];
      Bu[i] = y_A[i] * x_Bu[i] + y_Bu[i];
    }

    return result;
  }
};

// Functor to load a row from pointers (for use with zip_iterator)
// Works with a tuple of pointers from the zip_iterator
template <typename T, int HIDDEN_DIM>
struct LoadRowFromPointersFunctor
{
  template <typename Tuple>
  __host__ __device__ RowPair<T, HIDDEN_DIM> operator()(const Tuple& ptrs) const
  {
    RowPair<T, HIDDEN_DIM> result;
    auto& [a_arr, bu_arr] = result;
    auto a_ptr            = thrust::get<0>(ptrs);
    auto bu_ptr           = thrust::get<1>(ptrs);

#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      a_arr[i]  = a_ptr[i];
      bu_arr[i] = bu_ptr[i];
    }

    return result;
  }
};

// Functor to convert row index to pointer (for use with transform_iterator)
// Supports both const and non-const pointers
template <typename PtrType>
struct IndexToPointerFunctor
{
  PtrType base_ptr;
  int stride;

  __host__ __device__ IndexToPointerFunctor(PtrType ptr, int s)
      : base_ptr(ptr)
      , stride(s)
  {}

  __host__ __device__ PtrType operator()(int i) const
  {
    return base_ptr + i * stride;
  }
};

// Proxy reference type that only handles assignment for writing VectorPair results
template <typename T, int HIDDEN_DIM>
struct VectorPairWriteProxy
{
  T* a_row;
  T* bu_row;

  __host__ __device__ VectorPairWriteProxy& operator=(const RowPair<T, HIDDEN_DIM>& val)
  {
    auto& [A, Bu] = val;

#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      a_row[i]  = A[i];
      bu_row[i] = Bu[i];
    }
    return *this;
  }
};

// Functor to convert pointer tuple to write proxy (for use with transform_iterator)
// This reuses the IndexToPointerFunctor pattern for better composability
template <typename T, int HIDDEN_DIM>
struct PointersToWriteProxyFunctor
{
  template <typename Tuple>
  __host__ __device__ VectorPairWriteProxy<T, HIDDEN_DIM> operator()(const Tuple& ptrs) const
  {
    return {thrust::get<0>(ptrs), thrust::get<1>(ptrs)};
  }
};

// ============================================================================
// Helper to generate data (with special handling for __half)
// ============================================================================

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
  thrust::device_vector<T> d_A_out(timesteps * StateDim, thrust::no_init);
  thrust::device_vector<T> d_Bu_out(timesteps * StateDim, thrust::no_init);

  // This is a strided iterator. We can't use cuda::strided_iterator because it
  // returns elements, not pointers to rows.
  auto row_ptr_A = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    IndexToPointerFunctor<const T*>(thrust::raw_pointer_cast(d_A_in.data()), StateDim));

  auto row_ptr_Bu = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    IndexToPointerFunctor<const T*>(thrust::raw_pointer_cast(d_Bu_in.data()), StateDim));

  auto zipped_ptrs = thrust::make_zip_iterator(thrust::make_tuple(row_ptr_A, row_ptr_Bu));
  auto input_iter  = thrust::make_transform_iterator(zipped_ptrs, LoadRowFromPointersFunctor<T, StateDim>());

  auto row_ptr_A_out = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), IndexToPointerFunctor<T*>(thrust::raw_pointer_cast(d_A_out.data()), StateDim));

  auto row_ptr_Bu_out = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), IndexToPointerFunctor<T*>(thrust::raw_pointer_cast(d_Bu_out.data()), StateDim));

  auto zipped_ptrs_out = thrust::make_zip_iterator(thrust::make_tuple(row_ptr_A_out, row_ptr_Bu_out));
  auto output_iter     = thrust::make_transform_iterator(zipped_ptrs_out, PointersToWriteProxyFunctor<T, StateDim>());

  cub::DeviceScan::InclusiveScan(
    nullptr, temp_storage_bytes, input_iter, output_iter, S5Operator2D<T, StateDim>(), timesteps);

  return input_iter;
}

template <typename T, int StateDim, typename InputIterator>
void run_scan(
  void* d_temp_storage, size_t temp_storage_bytes, InputIterator input_iter, int timesteps, nvbench::launch& launch)
{
  thrust::device_vector<T> d_A_out_inner(timesteps * StateDim, thrust::no_init);
  thrust::device_vector<T> d_Bu_out_inner(timesteps * StateDim, thrust::no_init);

  auto row_ptr_A_out = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    IndexToPointerFunctor<T*>(thrust::raw_pointer_cast(d_A_out_inner.data()), StateDim));
  auto row_ptr_Bu_out = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    IndexToPointerFunctor<T*>(thrust::raw_pointer_cast(d_Bu_out_inner.data()), StateDim));
  auto zipped_ptrs_out = thrust::make_zip_iterator(thrust::make_tuple(row_ptr_A_out, row_ptr_Bu_out));
  auto output_iter     = thrust::make_transform_iterator(zipped_ptrs_out, PointersToWriteProxyFunctor<T, StateDim>());

  cub::DeviceScan::InclusiveScan(
    d_temp_storage,
    temp_storage_bytes,
    input_iter,
    output_iter,
    S5Operator2D<T, StateDim>(),
    timesteps,
    launch.get_stream());
}
