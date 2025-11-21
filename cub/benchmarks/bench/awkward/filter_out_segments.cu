#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <nvbench_helper.cuh>

#include "filter_out_segments_copy_if.cuh"
#include "filter_out_segments_rle_scan.cuh"

template <typename T>
static void print_array(const thrust::device_vector<T>& d_values, const thrust::device_vector<int>& d_offsets)
{
  thrust::host_vector<T> h_values    = d_values;
  thrust::host_vector<int> h_offsets = d_offsets;

  int num_segments = static_cast<int>(h_offsets.size()) - 1;

  for (int seg = 0; seg < num_segments; ++seg)
  {
    int start = h_offsets[seg];
    int end   = h_offsets[seg + 1];

    std::cout << "Segment " << seg << ": ";
    for (int i = start; i < end; ++i)
    {
      std::cout << h_values[i] << " ";
    }
    std::cout << std::endl;
  }
}

// Step 1: Implementation using segment ids, run-length encoding, and a scan
template <typename T>
static void filter_out_segments_rle_scan(nvbench::state& state, nvbench::type_list<T>)
{
  // Example: [[30], [40,20], [50], [10,30,80]]
  // Flatten:
  thrust::device_vector<T> d_values{30, 40, 20, 50, 10, 30, 80};
  thrust::device_vector<int> d_offsets{0, 1, 3, 4, 7}; // 4 segments
  thrust::device_vector<bool> d_mask{true, false, false, true}; // Keep segments 0 and 3
  std::cout << "Before filtering:" << std::endl;
  print_array(d_values, d_offsets);

  filter_out_segments_rle_scan(d_values, d_offsets, d_mask);

  std::cout << "After filtering:" << std::endl;
  print_array(d_values, d_offsets);
}

// Step 2: Implement using segment ids and cub::DeviceSelect::If
template <typename T>
static void filter_out_segments_copy_if(nvbench::state& state, nvbench::type_list<T>)
{
  // Example: [[30], [40,20], [50], [10,30,80]]
  // Flatten:
  thrust::device_vector<T> d_values{30, 40, 20, 50, 10, 30, 80};
  thrust::device_vector<int> d_offsets{0, 1, 3, 4, 7}; // 4 segments
  thrust::device_vector<bool> d_mask{true, false, false, true}; // Keep segments 0 and 3
  std::cout << "Before filtering:" << std::endl;
  print_array(d_values, d_offsets);

  filter_out_segments_copy_if(d_values, d_offsets, d_mask);

  std::cout << "After filtering:" << std::endl;
  print_array(d_values, d_offsets);
}

using current_data_types = nvbench::type_list<float>;

NVBENCH_BENCH_TYPES(filter_out_segments_rle_scan, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("filter_out_segments_rle_scan")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 12, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});

NVBENCH_BENCH_TYPES(filter_out_segments_copy_if, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("filter_out_segments_copy_if")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 12, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
