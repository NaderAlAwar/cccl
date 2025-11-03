// S5 Associative Scan Verification Tool
// Reads input from .npy files, runs scan, writes output to .npy files for comparison

#include <cuda_fp16.h>

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstring>
#include <iostream>
#include <string>

// Include libnpy
#include "libnpy/include/npy.hpp"

// Reuse operators and functors from the benchmark code
// S5 operator working on thrust::tuple
template <typename T>
struct S5Operator
{
  using Tuple = thrust::tuple<T, T>;

  __host__ __device__ Tuple operator()(const Tuple& x, const Tuple& y) const
  {
    return thrust::make_tuple(thrust::get<0>(y) * thrust::get<0>(x),
                              thrust::get<0>(y) * thrust::get<1>(x) + thrust::get<1>(y));
  }
};

// VectorPair for 2D case
template <typename T, int HIDDEN_DIM>
struct VectorPair
{
  T A[HIDDEN_DIM];
  T Bu[HIDDEN_DIM];

  __host__ __device__ VectorPair() = default;

  __host__ __device__ VectorPair(const T* a_ptr, const T* bu_ptr)
  {
#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      A[i]  = a_ptr[i];
      Bu[i] = bu_ptr[i];
    }
  }
};

// S5 operator for 2D tensors
template <typename T, int HIDDEN_DIM>
struct S5Operator2D
{
  __host__ __device__ VectorPair<T, HIDDEN_DIM>
  operator()(const VectorPair<T, HIDDEN_DIM>& x, const VectorPair<T, HIDDEN_DIM>& y) const
  {
    VectorPair<T, HIDDEN_DIM> result;
#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      result.A[i]  = y.A[i] * x.A[i];
      result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i];
    }
    return result;
  }
};

// Functors for 2D iterators
template <typename T>
struct IndexToPointerFunctor
{
  const T* base_ptr;
  int stride;

  __host__ __device__ IndexToPointerFunctor(const T* ptr, int s)
      : base_ptr(ptr)
      , stride(s)
  {}

  __host__ __device__ const T* operator()(int i) const
  {
    return base_ptr + i * stride;
  }
};

template <typename T, int HIDDEN_DIM>
struct LoadRowFromPointersFunctor
{
  template <typename Tuple>
  __host__ __device__ VectorPair<T, HIDDEN_DIM> operator()(const Tuple& ptrs) const
  {
    return VectorPair<T, HIDDEN_DIM>(thrust::get<0>(ptrs), thrust::get<1>(ptrs));
  }
};

template <typename T, int HIDDEN_DIM>
struct VectorPairOutputIterator
{
  T* A_base;
  T* Bu_base;
  int stride;
  int current_idx;

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = VectorPair<T, HIDDEN_DIM>;
  using difference_type   = ptrdiff_t;
  using pointer           = value_type*;
  using reference         = VectorPairOutputIterator&;

  __host__ __device__ VectorPairOutputIterator(T* a, T* bu, int s, int idx = 0)
      : A_base(a)
      , Bu_base(bu)
      , stride(s)
      , current_idx(idx)
  {}

  __host__ __device__ reference operator[](difference_type n)
  {
    return *(VectorPairOutputIterator(A_base, Bu_base, stride, current_idx + n));
  }

  __host__ __device__ reference operator*()
  {
    return *this;
  }

  __host__ __device__ VectorPairOutputIterator& operator=(const VectorPair<T, HIDDEN_DIM>& val)
  {
    T* a_row  = A_base + current_idx * stride;
    T* bu_row = Bu_base + current_idx * stride;
#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      a_row[i]  = val.A[i];
      bu_row[i] = val.Bu[i];
    }
    return *this;
  }

  __host__ __device__ VectorPairOutputIterator operator+(difference_type n) const
  {
    return VectorPairOutputIterator(A_base, Bu_base, stride, current_idx + n);
  }

  __host__ __device__ difference_type operator-(const VectorPairOutputIterator& other) const
  {
    return current_idx - other.current_idx;
  }
};

// Helper to convert __half to/from float for npy I/O
template <typename T>
void read_npy_file(const std::string& filename, std::vector<T>& data)
{
  if constexpr (cuda::std::is_same_v<T, __half>)
  {
    // Read as float, then convert to __half
    npy::npy_data<float> npy_data = npy::read_npy<float>(filename);
    data.resize(npy_data.data.size());
    for (size_t i = 0; i < npy_data.data.size(); i++)
    {
      data[i] = __float2half(npy_data.data[i]);
    }
  }
  else
  {
    npy::npy_data<T> npy_data = npy::read_npy<T>(filename);
    data                      = npy_data.data;
  }
}

template <typename T>
void write_npy_file(const std::string& filename, const std::vector<T>& data, const std::vector<unsigned long>& shape)
{
  if constexpr (cuda::std::is_same_v<T, __half>)
  {
    // Convert __half to float, then write
    std::vector<float> float_data(data.size());
    for (size_t i = 0; i < data.size(); i++)
    {
      float_data[i] = __half2float(data[i]);
    }
    npy::npy_data<float> npy_data;
    npy_data.data          = float_data;
    npy_data.shape         = shape;
    npy_data.fortran_order = false;
    npy::write_npy(filename, npy_data);
  }
  else
  {
    npy::npy_data<T> npy_data;
    npy_data.data          = data;
    npy_data.shape         = shape;
    npy_data.fortran_order = false;
    npy::write_npy(filename, npy_data);
  }
}

template <typename T>
void run_verification_1d(int timesteps)
{
  std::cout << "Running 1D verification (timesteps=" << timesteps << ")" << std::endl;

  // Read input data
  std::vector<T> h_A_in, h_Bu_in;
  read_npy_file<T>("A_in.npy", h_A_in);
  read_npy_file<T>("Bu_in.npy", h_Bu_in);

  if (h_A_in.size() != timesteps || h_Bu_in.size() != timesteps)
  {
    std::cerr << "Error: Input size mismatch. Expected " << timesteps << " but got A=" << h_A_in.size()
              << ", Bu=" << h_Bu_in.size() << std::endl;
    return;
  }

  // Copy to device
  thrust::device_vector<T> d_A_in(h_A_in);
  thrust::device_vector<T> d_Bu_in(h_Bu_in);
  thrust::device_vector<T> d_A_out(timesteps);
  thrust::device_vector<T> d_Bu_out(timesteps);

  // Run scan
  auto input_iter  = thrust::make_zip_iterator(thrust::make_tuple(d_A_in.begin(), d_Bu_in.begin()));
  auto output_iter = thrust::make_zip_iterator(thrust::make_tuple(d_A_out.begin(), d_Bu_out.begin()));

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator<T>(), timesteps);

  thrust::device_vector<uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator<T>(), timesteps);

  cudaDeviceSynchronize();

  // Copy results back
  thrust::host_vector<T> h_A_out  = d_A_out;
  thrust::host_vector<T> h_Bu_out = d_Bu_out;

  // Convert to std::vector for npy write
  std::vector<T> A_out_vec(h_A_out.begin(), h_A_out.end());
  std::vector<T> Bu_out_vec(h_Bu_out.begin(), h_Bu_out.end());

  // Write output
  write_npy_file<T>("A_out_cpp.npy", A_out_vec, {static_cast<unsigned long>(timesteps)});
  write_npy_file<T>("Bu_out_cpp.npy", Bu_out_vec, {static_cast<unsigned long>(timesteps)});

  std::cout << "Output written to A_out_cpp.npy and Bu_out_cpp.npy" << std::endl;
}

template <typename T, int STATE_DIM>
void run_verification_2d(int timesteps)
{
  std::cout << "Running 2D verification (timesteps=" << timesteps << ", state_dim=" << STATE_DIM << ")" << std::endl;

  int total_size = timesteps * STATE_DIM;

  // Read input data
  std::vector<T> h_A_in, h_Bu_in;
  read_npy_file<T>("A_in.npy", h_A_in);
  read_npy_file<T>("Bu_in.npy", h_Bu_in);

  if (h_A_in.size() != total_size || h_Bu_in.size() != total_size)
  {
    std::cerr << "Error: Input size mismatch. Expected " << total_size << " but got A=" << h_A_in.size()
              << ", Bu=" << h_Bu_in.size() << std::endl;
    return;
  }

  // Copy to device
  thrust::device_vector<T> d_A_in(h_A_in);
  thrust::device_vector<T> d_Bu_in(h_Bu_in);
  thrust::device_vector<T> d_A_out(total_size);
  thrust::device_vector<T> d_Bu_out(total_size);

  // Create iterators
  auto row_ptr_A = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), IndexToPointerFunctor<T>(thrust::raw_pointer_cast(d_A_in.data()), STATE_DIM));

  auto row_ptr_Bu = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), IndexToPointerFunctor<T>(thrust::raw_pointer_cast(d_Bu_in.data()), STATE_DIM));

  auto zipped_ptrs = thrust::make_zip_iterator(thrust::make_tuple(row_ptr_A, row_ptr_Bu));
  auto input_iter  = thrust::make_transform_iterator(zipped_ptrs, LoadRowFromPointersFunctor<T, STATE_DIM>());
  auto output_iter = VectorPairOutputIterator<T, STATE_DIM>(
    thrust::raw_pointer_cast(d_A_out.data()), thrust::raw_pointer_cast(d_Bu_out.data()), STATE_DIM);

  // Run scan
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator2D<T, STATE_DIM>(), timesteps);

  thrust::device_vector<uint8_t> d_temp(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator2D<T, STATE_DIM>(), timesteps);

  cudaDeviceSynchronize();

  // Copy results back
  thrust::host_vector<T> h_A_out  = d_A_out;
  thrust::host_vector<T> h_Bu_out = d_Bu_out;

  // Convert to std::vector for npy write
  std::vector<T> A_out_vec(h_A_out.begin(), h_A_out.end());
  std::vector<T> Bu_out_vec(h_Bu_out.begin(), h_Bu_out.end());

  // Write output
  write_npy_file<T>(
    "A_out_cpp.npy", A_out_vec, {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(STATE_DIM)});
  write_npy_file<T>(
    "Bu_out_cpp.npy", Bu_out_vec, {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(STATE_DIM)});

  std::cout << "Output written to A_out_cpp.npy and Bu_out_cpp.npy" << std::endl;
}

void print_usage()
{
  std::cout << "Usage: s5_verify --timesteps N --is-2d [0|1] --dtype [float16|float32|float64]" << std::endl;
  std::cout << "  --timesteps: Number of timesteps (sequence length)" << std::endl;
  std::cout << "  --is-2d: 1 for 2D (vector) scan, 0 for 1D (scalar) scan" << std::endl;
  std::cout << "  --dtype: Data type (float16, float32, or float64)" << std::endl;
}

int main(int argc, char** argv)
{
  // Parse command line arguments
  int timesteps           = 0;
  int is_2d               = -1;
  std::string dtype       = "";
  constexpr int state_dim = 40;

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--timesteps") == 0 && i + 1 < argc)
    {
      timesteps = std::atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--is-2d") == 0 && i + 1 < argc)
    {
      is_2d = std::atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc)
    {
      dtype = argv[++i];
    }
    else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
    {
      print_usage();
      return 0;
    }
  }

  if (timesteps <= 0 || is_2d < 0 || dtype.empty())
  {
    std::cerr << "Error: Missing required arguments" << std::endl;
    print_usage();
    return 1;
  }

  std::cout << "S5 Scan Verification" << std::endl;
  std::cout << "  Timesteps: " << timesteps << std::endl;
  std::cout << "  Mode: " << (is_2d ? "2D" : "1D") << std::endl;
  std::cout << "  Dtype: " << dtype << std::endl;

  // Run verification based on dtype and mode
  if (dtype == "float16")
  {
    if (is_2d)
    {
      run_verification_2d<__half, state_dim>(timesteps);
    }
    else
    {
      run_verification_1d<__half>(timesteps);
    }
  }
  else if (dtype == "float32")
  {
    if (is_2d)
    {
      run_verification_2d<float, state_dim>(timesteps);
    }
    else
    {
      run_verification_1d<float>(timesteps);
    }
  }
  else if (dtype == "float64")
  {
    if (is_2d)
    {
      run_verification_2d<double, state_dim>(timesteps);
    }
    else
    {
      run_verification_1d<double>(timesteps);
    }
  }
  else
  {
    std::cerr << "Error: Unknown dtype: " << dtype << std::endl;
    return 1;
  }

  return 0;
}
