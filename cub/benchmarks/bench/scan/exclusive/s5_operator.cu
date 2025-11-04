// CUB S5 Associative Scan Benchmark
// Demonstrates 1D (scalar) and 2D (vector/row-wise) associative scans

#include <cuda_fp16.h>

#include <cub/device/device_scan.cuh>

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

// ============================================================================
// 2D Tensor Version - Elementwise Operations (Using Thrust/CUDA utilities)
// ============================================================================
//
// COMPARISON: 1D vs 2D Approaches
// --------------------------------
//
// 1D Case (scalar elements):
//   - Elements: float values
//   - Iterator: thrust::zip_iterator(thrust::make_tuple(A_ptr, Bu_ptr))
//   - Operator: Works on thrust::tuple<float, float>
//
// 2D Case (vector elements):
//   - Elements: VectorPair (contains float arrays)
//   - Iterator: zip_iterator + transform_iterator composition
//   - Operator: Works on VectorPair<HIDDEN_DIM> with elementwise ops
//
// Iterator Composition Strategy:
// --------------------------------
//   counting_iterator
//     -> transform to A_row_pointers (IndexToPointerFunctor)
//   counting_iterator
//     -> transform to Bu_row_pointers (IndexToPointerFunctor)
//   zip_iterator(A_pointers, Bu_pointers)
//     -> tuple of pointers
//   transform_iterator (LoadRowFromPointersFunctor)
//     -> VectorPair objects
//
// Key Thrust/CUDA utilities used:
// --------------------------------
// 1. thrust::transform_iterator - wraps an iterator and applies a function on dereference
// 2. thrust::counting_iterator - generates a sequence of integers (0, 1, 2, ...)
// 3. thrust::zip_iterator - combines multiple iterators into tuples
//
// This approach is composable and demonstrates how to use standard Thrust
// iterators instead of writing full custom iterators from scratch.
// We only implement minimal functor logic and let Thrust handle iterator machinery.
// ============================================================================

// Helper struct to represent a vector at a specific position in the 2D tensor
// This stores the actual data (not a view), which is necessary for scan operations
// that need to materialize intermediate results.
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

// S5 operator for 2D tensors with elementwise operations
template <typename T, int HIDDEN_DIM>
struct S5Operator2D
{
  __host__ __device__ VectorPair<T, HIDDEN_DIM>
  operator()(const VectorPair<T, HIDDEN_DIM>& x, const VectorPair<T, HIDDEN_DIM>& y) const
  {
    VectorPair<T, HIDDEN_DIM> result;

// Elementwise operations:
// result.A[i] = y.A[i] * x.A[i]
// result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i]
#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      result.A[i]  = y.A[i] * x.A[i];
      result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i];
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
  __host__ __device__ VectorPair<T, HIDDEN_DIM> operator()(const Tuple& ptrs) const
  {
    return VectorPair<T, HIDDEN_DIM>(thrust::get<0>(ptrs), thrust::get<1>(ptrs));
  }
};

// Functor to convert row index to pointer (for use with transform_iterator)
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

// Proxy reference type that only handles assignment for writing VectorPair results
template <typename T, int HIDDEN_DIM>
struct VectorPairWriteProxy
{
  T* a_row;
  T* bu_row;

  __host__ __device__ VectorPairWriteProxy& operator=(const VectorPair<T, HIDDEN_DIM>& val)
  {
#pragma unroll
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      a_row[i]  = val.A[i];
      bu_row[i] = val.Bu[i];
    }
    return *this;
  }
};

// Functor to convert row index to write proxy (for use with transform_iterator)
template <typename T, int HIDDEN_DIM>
struct IndexToWriteProxyFunctor
{
  T* A_base;
  T* Bu_base;
  int stride;

  __host__ __device__ IndexToWriteProxyFunctor(T* a, T* bu, int s)
      : A_base(a)
      , Bu_base(bu)
      , stride(s)
  {}

  __host__ __device__ VectorPairWriteProxy<T, HIDDEN_DIM> operator()(int idx) const
  {
    return {A_base + idx * stride, Bu_base + idx * stride};
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

// ============================================================================
// NVBench Benchmark
// ============================================================================

template <typename T>
static void s5_scan_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // SSM/S5 typical parameters
  const int timesteps     = state.get_int64("Timesteps"); // sequence length
  constexpr int state_dim = 40; // compile-time hidden dimension

  // Get benchmark parameter
  const bool is_2d = state.get_int64("is_2d");

  if (is_2d)
  {
    // 2D case: vector elements with elementwise operations
    int total_size = timesteps * state_dim;

    // Allocate and initialize data using nvbench_helper
    thrust::device_vector<T> d_A_in  = generate_data<T>(total_size);
    thrust::device_vector<T> d_Bu_in = generate_data<T>(total_size);
    thrust::device_vector<T> d_A_out(total_size, thrust::no_init);
    thrust::device_vector<T> d_Bu_out(total_size, thrust::no_init);

    // Pre-allocate temporary storage
    auto row_ptr_A = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToPointerFunctor<T>(thrust::raw_pointer_cast(d_A_in.data()), state_dim));

    auto row_ptr_Bu = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToPointerFunctor<T>(thrust::raw_pointer_cast(d_Bu_in.data()), state_dim));

    auto zipped_ptrs = thrust::make_zip_iterator(thrust::make_tuple(row_ptr_A, row_ptr_Bu));

    auto input_iter = thrust::make_transform_iterator(zipped_ptrs, LoadRowFromPointersFunctor<T, state_dim>());

    auto output_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      IndexToWriteProxyFunctor<T, state_dim>(
        thrust::raw_pointer_cast(d_A_out.data()), thrust::raw_pointer_cast(d_Bu_out.data()), state_dim));

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator2D<T, state_dim>(), timesteps);

    thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
    d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

    state.add_element_count(timesteps);
    state.add_global_memory_reads<T>(total_size, "A_in");
    state.add_global_memory_reads<T>(total_size, "Bu_in");
    state.add_global_memory_writes<T>(total_size, "A_out");
    state.add_global_memory_writes<T>(total_size, "Bu_out");

    // Benchmark the scan operation
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      // Move this stuff here for fair comparison with pytorch
      thrust::device_vector<T> d_A_out_inner(total_size, thrust::no_init);
      thrust::device_vector<T> d_Bu_out_inner(total_size, thrust::no_init);
      auto output_iter_inner = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        IndexToWriteProxyFunctor<T, state_dim>(
          thrust::raw_pointer_cast(d_A_out_inner.data()), thrust::raw_pointer_cast(d_Bu_out_inner.data()), state_dim));
      cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        input_iter,
        output_iter_inner,
        S5Operator2D<T, state_dim>(),
        timesteps,
        launch.get_stream());
    });
  }
  else
  {
    // 1D case: scalar elements
    thrust::device_vector<T> d_A_in  = generate_data<T>(timesteps);
    thrust::device_vector<T> d_Bu_in = generate_data<T>(timesteps);
    thrust::device_vector<T> d_A_out(timesteps, thrust::no_init);
    thrust::device_vector<T> d_Bu_out(timesteps, thrust::no_init);

    // Pre-allocate temporary storage
    auto input_iter  = thrust::make_zip_iterator(thrust::make_tuple(d_A_in.begin(), d_Bu_in.begin()));
    auto output_iter = thrust::make_zip_iterator(thrust::make_tuple(d_A_out.begin(), d_Bu_out.begin()));

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, input_iter, output_iter, S5Operator<T>(), timesteps);

    thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
    d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

    state.add_element_count(timesteps);
    state.add_global_memory_reads<T>(timesteps, "A_in");
    state.add_global_memory_reads<T>(timesteps, "Bu_in");
    state.add_global_memory_writes<T>(timesteps, "A_out");
    state.add_global_memory_writes<T>(timesteps, "Bu_out");

    // Benchmark the scan operation
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      // Move this stuff here for fair comparison with pytorch
      thrust::device_vector<T> d_A_out_inner(timesteps, thrust::no_init);
      thrust::device_vector<T> d_Bu_out_inner(timesteps, thrust::no_init);
      auto output_iter_inner =
        thrust::make_zip_iterator(thrust::make_tuple(d_A_out_inner.begin(), d_Bu_out_inner.begin()));
      cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        input_iter,
        output_iter_inner,
        S5Operator<T>(),
        timesteps,
        launch.get_stream());
    });
  }
}

using data_types = nvbench::type_list<__half, float, double>;

NVBENCH_BENCH_TYPES(s5_scan_benchmark, NVBENCH_TYPE_AXES(data_types))
  .set_name("s5_associative_scan")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("is_2d", {1})
  .add_int64_power_of_two_axis("Timesteps", nvbench::range(16, 24, 4));
