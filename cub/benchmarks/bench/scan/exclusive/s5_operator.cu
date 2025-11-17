// CUB S5 Associative Scan Benchmark
// Demonstrates 1D (scalar) and 2D (vector/row-wise) associative scans

#include <cuda_fp16.h>

#include <cub/device/device_scan.cuh>

#include <thrust/host_vector.h>

#include <cuda/iterator>

#include <nvbench_helper.cuh>

#include "s5_operator.cuh"

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
    thrust::device_vector<T> d_A_in  = s5_operator::generate_data<T>(total_size);
    thrust::device_vector<T> d_Bu_in = s5_operator::generate_data<T>(total_size);
    thrust::device_vector<T> d_A_out(total_size, thrust::no_init);
    thrust::device_vector<T> d_Bu_out(total_size, thrust::no_init);

    size_t temp_storage_bytes = 0;
    auto input_iter           = s5_operator::setup_scan<T, state_dim>(d_A_in, d_Bu_in, timesteps, temp_storage_bytes);

    thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
    void* d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

    state.add_element_count(timesteps);

    // Benchmark the scan operation
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::device_vector<T> d_A_out_inner(timesteps * state_dim, thrust::no_init);
      thrust::device_vector<T> d_Bu_out_inner(timesteps * state_dim, thrust::no_init);

      cudaStream_t stream = launch.get_stream();

      s5_operator::run_scan<T, state_dim>(
        d_temp_storage, temp_storage_bytes, input_iter, d_A_out_inner, d_Bu_out_inner, timesteps, stream);
    });
  }
  else
  {
    // 1D case: scalar elements
    thrust::device_vector<T> d_A_in  = s5_operator::generate_data<T>(timesteps);
    thrust::device_vector<T> d_Bu_in = s5_operator::generate_data<T>(timesteps);
    thrust::device_vector<T> d_A_out(timesteps, thrust::no_init);
    thrust::device_vector<T> d_Bu_out(timesteps, thrust::no_init);

    // Pre-allocate temporary storage
    auto input_iter  = thrust::make_zip_iterator(thrust::make_tuple(d_A_in.begin(), d_Bu_in.begin()));
    auto output_iter = thrust::make_zip_iterator(thrust::make_tuple(d_A_out.begin(), d_Bu_out.begin()));

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, input_iter, output_iter, s5_operator::S5Operator<T>(), timesteps);

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
        s5_operator::S5Operator<T>(),
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
  .add_int64_power_of_two_axis("Timesteps", nvbench::range(12, 24, 2));
