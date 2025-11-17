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

#include "s5_operator_segmented.cuh"

template <typename T>
void s5_scan_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  const int timesteps = static_cast<int>(state.get_int64("Timesteps"));
  const int state_dim = 40;
  const bool is_2d    = state.get_int64("is_2d") != 0;

  const int total_elements = timesteps * state_dim;

  // Allocate and initialize data using nvbench_helper
  thrust::device_vector<T> d_A_in  = s5_operator_segmented::generate_data<T>(total_elements);
  thrust::device_vector<T> d_Bu_in = s5_operator_segmented::generate_data<T>(total_elements);

  state.add_element_count(timesteps * state_dim);

  size_t temp_storage_bytes = 0;
  auto input_iter = s5_operator_segmented::setup_scan<T, state_dim>(d_A_in, d_Bu_in, timesteps, temp_storage_bytes);

  thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    thrust::device_vector<T> d_A_out(total_elements, thrust::no_init);
    thrust::device_vector<T> d_Bu_out(total_elements, thrust::no_init);

    cudaStream_t stream = launch.get_stream();

    s5_operator_segmented::run_scan<T, state_dim>(
      d_temp_storage, temp_storage_bytes, input_iter, d_A_out, d_Bu_out, timesteps, stream);
  });
}

using data_types = nvbench::type_list<__half, float, double>;

NVBENCH_BENCH_TYPES(s5_scan_benchmark, NVBENCH_TYPE_AXES(data_types))
  .set_name("s5_associative_scan")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("is_2d", {1})
  .add_int64_power_of_two_axis("Timesteps", nvbench::range(16, 24, 4));
