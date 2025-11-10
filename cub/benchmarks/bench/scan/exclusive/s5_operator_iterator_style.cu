// CUB S5 Scan - Iterator Style (Matching cuda.compute pattern)
// This version mimics the cuda.compute implementation pattern for direct comparison

#include <cuda_fp16.h>

#include <cub/device/device_scan.cuh>

#include <nvbench_helper.cuh>

// ============================================================================
// Custom Iterator Pattern (Mimicking DualRowPointerIterator)
// ============================================================================

// Value type: nested tuple structure matching cuda.compute
// ((A0, A1, A2, A3), (Bu0, Bu1, Bu2, Bu3))
template <typename T, int STATE_DIM>
struct VectorPairValue
{
  T A[STATE_DIM];
  T Bu[STATE_DIM];

  __host__ __device__ VectorPairValue() = default;

  __host__ __device__ VectorPairValue(const T* a_ptr, const T* bu_ptr)
  {
#pragma unroll
    for (int i = 0; i < STATE_DIM; i++)
    {
      A[i]  = a_ptr[i];
      Bu[i] = bu_ptr[i];
    }
  }
};

// Custom iterator that stores row pointers and dereferences on access
// This mimics DualRowPointerIterator behavior
template <typename T, int STATE_DIM>
class DualRowPointerIterator
{
public:
  // Iterator traits
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = VectorPairValue<T, STATE_DIM>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = value_type*;
  using reference         = value_type;

private:
  const T* a_base_ptr_;
  const T* bu_base_ptr_;
  int row_stride_; // Elements per row
  difference_type row_index_;

public:
  __host__ __device__ DualRowPointerIterator()
      : a_base_ptr_(nullptr)
      , bu_base_ptr_(nullptr)
      , row_stride_(0)
      , row_index_(0)
  {}

  __host__ __device__ DualRowPointerIterator(const T* a_ptr, const T* bu_ptr, int stride, difference_type idx = 0)
      : a_base_ptr_(a_ptr)
      , bu_base_ptr_(bu_ptr)
      , row_stride_(stride)
      , row_index_(idx)
  {}

  // Dereference: load row values into VectorPairValue
  // This is where the actual data loading happens (mimics input_dereference_impl)
  __device__ reference operator*() const
  {
    const T* a_row  = a_base_ptr_ + row_index_ * row_stride_;
    const T* bu_row = bu_base_ptr_ + row_index_ * row_stride_;
    return VectorPairValue<T, STATE_DIM>(a_row, bu_row);
  }

  __device__ reference operator[](difference_type n) const
  {
    const T* a_row  = a_base_ptr_ + (row_index_ + n) * row_stride_;
    const T* bu_row = bu_base_ptr_ + (row_index_ + n) * row_stride_;
    return VectorPairValue<T, STATE_DIM>(a_row, bu_row);
  }

  // Iterator arithmetic
  __host__ __device__ DualRowPointerIterator& operator+=(difference_type n)
  {
    row_index_ += n;
    return *this;
  }

  __host__ __device__ DualRowPointerIterator& operator-=(difference_type n)
  {
    row_index_ -= n;
    return *this;
  }

  __host__ __device__ DualRowPointerIterator operator+(difference_type n) const
  {
    return DualRowPointerIterator(a_base_ptr_, bu_base_ptr_, row_stride_, row_index_ + n);
  }

  __host__ __device__ DualRowPointerIterator operator-(difference_type n) const
  {
    return DualRowPointerIterator(a_base_ptr_, bu_base_ptr_, row_stride_, row_index_ - n);
  }

  __host__ __device__ difference_type operator-(const DualRowPointerIterator& other) const
  {
    return row_index_ - other.row_index_;
  }

  __host__ __device__ bool operator==(const DualRowPointerIterator& other) const
  {
    return row_index_ == other.row_index_;
  }

  __host__ __device__ bool operator!=(const DualRowPointerIterator& other) const
  {
    return row_index_ != other.row_index_;
  }
};

// Output iterator that writes VectorPairValue back to separate arrays
// This mimics output_dereference_impl behavior
template <typename T, int STATE_DIM>
class DualRowPointerOutputIterator
{
public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = VectorPairValue<T, STATE_DIM>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = void;

  // Proxy reference for assignment
  struct Reference
  {
    T* a_row_;
    T* bu_row_;

    __device__ Reference(T* a_row, T* bu_row)
        : a_row_(a_row)
        , bu_row_(bu_row)
    {}

    __device__ Reference& operator=(const VectorPairValue<T, STATE_DIM>& val)
    {
#pragma unroll
      for (int i = 0; i < STATE_DIM; i++)
      {
        a_row_[i]  = val.A[i];
        bu_row_[i] = val.Bu[i];
      }
      return *this;
    }
  };

  using reference = Reference;

private:
  T* a_base_ptr_;
  T* bu_base_ptr_;
  int row_stride_;
  difference_type row_index_;

public:
  __host__ __device__ DualRowPointerOutputIterator()
      : a_base_ptr_(nullptr)
      , bu_base_ptr_(nullptr)
      , row_stride_(0)
      , row_index_(0)
  {}

  __host__ __device__ DualRowPointerOutputIterator(T* a_ptr, T* bu_ptr, int stride, difference_type idx = 0)
      : a_base_ptr_(a_ptr)
      , bu_base_ptr_(bu_ptr)
      , row_stride_(stride)
      , row_index_(idx)
  {}

  __device__ Reference operator*() const
  {
    T* a_row  = a_base_ptr_ + row_index_ * row_stride_;
    T* bu_row = bu_base_ptr_ + row_index_ * row_stride_;
    return Reference(a_row, bu_row);
  }

  __device__ Reference operator[](difference_type n) const
  {
    T* a_row  = a_base_ptr_ + (row_index_ + n) * row_stride_;
    T* bu_row = bu_base_ptr_ + (row_index_ + n) * row_stride_;
    return Reference(a_row, bu_row);
  }

  __host__ __device__ DualRowPointerOutputIterator& operator+=(difference_type n)
  {
    row_index_ += n;
    return *this;
  }

  __host__ __device__ DualRowPointerOutputIterator& operator-=(difference_type n)
  {
    row_index_ -= n;
    return *this;
  }

  __host__ __device__ DualRowPointerOutputIterator operator+(difference_type n) const
  {
    return DualRowPointerOutputIterator(a_base_ptr_, bu_base_ptr_, row_stride_, row_index_ + n);
  }

  __host__ __device__ DualRowPointerOutputIterator operator-(difference_type n) const
  {
    return DualRowPointerOutputIterator(a_base_ptr_, bu_base_ptr_, row_stride_, row_index_ - n);
  }

  __host__ __device__ difference_type operator-(const DualRowPointerOutputIterator& other) const
  {
    return row_index_ - other.row_index_;
  }
};

// ============================================================================
// S5 Operator on VectorPairValue (Matching cuda.compute s5_op)
// ============================================================================

template <typename T, int STATE_DIM>
struct S5OperatorIteratorStyle
{
  __host__ __device__ VectorPairValue<T, STATE_DIM>
  operator()(VectorPairValue<T, STATE_DIM> x, VectorPairValue<T, STATE_DIM> y) const
  {
    VectorPairValue<T, STATE_DIM> result;

    // Elementwise S5 operations:
    // result.A[i] = y.A[i] * x.A[i]
    // result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i]
#pragma unroll
    for (int i = 0; i < STATE_DIM; i++)
    {
      result.A[i]  = y.A[i] * x.A[i];
      result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i];
    }

    return result;
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

// ============================================================================
// NVBench Benchmark
// ============================================================================

template <typename T>
static void s5_scan_iterator_style_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  const int timesteps = state.get_int64("Timesteps");
  const int state_dim = state.get_int64("StateDim");

  const int total_size = timesteps * state_dim;

  // Allocate and initialize data
  thrust::device_vector<T> d_A_in  = generate_data<T>(total_size);
  thrust::device_vector<T> d_Bu_in = generate_data<T>(total_size);
  thrust::device_vector<T> d_A_out(total_size, thrust::no_init);
  thrust::device_vector<T> d_Bu_out(total_size, thrust::no_init);

  // Create iterators following the cuda.compute pattern
  auto input_iter = DualRowPointerIterator<T, 4>(
    thrust::raw_pointer_cast(d_A_in.data()), thrust::raw_pointer_cast(d_Bu_in.data()), state_dim);

  auto output_iter = DualRowPointerOutputIterator<T, 4>(
    thrust::raw_pointer_cast(d_A_out.data()), thrust::raw_pointer_cast(d_Bu_out.data()), state_dim);

  // Pre-allocate temporary storage
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5OperatorIteratorStyle<T, 4>(), timesteps);

  thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  state.add_element_count(timesteps);
  state.add_global_memory_reads<T>(total_size, "A_in");
  state.add_global_memory_reads<T>(total_size, "Bu_in");
  state.add_global_memory_writes<T>(total_size, "A_out");
  state.add_global_memory_writes<T>(total_size, "Bu_out");

  // Benchmark the scan operation
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate output inside for fair comparison
    thrust::device_vector<T> d_A_out_inner(total_size, thrust::no_init);
    thrust::device_vector<T> d_Bu_out_inner(total_size, thrust::no_init);

    auto output_iter_inner = DualRowPointerOutputIterator<T, 4>(
      thrust::raw_pointer_cast(d_A_out_inner.data()), thrust::raw_pointer_cast(d_Bu_out_inner.data()), state_dim);

    cub::DeviceScan::InclusiveScan(
      d_temp_storage,
      temp_storage_bytes,
      input_iter,
      output_iter_inner,
      S5OperatorIteratorStyle<T, 4>(),
      timesteps,
      launch.get_stream());
  });
}

using data_types = nvbench::type_list<float, double>;

NVBENCH_BENCH_TYPES(s5_scan_iterator_style_benchmark, NVBENCH_TYPE_AXES(data_types))
  .set_name("s5_iterator_style")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("StateDim", {4})
  .add_int64_power_of_two_axis("Timesteps", nvbench::range(16, 24, 4));
