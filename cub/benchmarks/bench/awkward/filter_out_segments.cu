#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <nvbench_helper.cuh>

#include "bench_util.cuh"
#include "filter_out_segments_copy_if.cuh"
#include "filter_out_segments_fancy_iterator.cuh"
#include "filter_out_segments_rle_scan.cuh"

// Step 1: Implementation using segment ids, run-length encoding, and a scan
template <typename T>
static void filter_out_segments_rle_scan(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
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
#else
  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  constexpr int max_segment_size = 20;
  const int max_num_segments     = cuda::ceil_div(elements, max_segment_size);

  thrust::device_vector<T> d_values = generate(elements);

  auto segment_sizes = partition_into_segments(elements, max_segment_size);
  thrust::host_vector<int> h_offsets(segment_sizes.size() + 1);
  std::exclusive_scan(segment_sizes.begin(), segment_sizes.end(), h_offsets.begin(), 0);
  h_offsets[segment_sizes.size()] = h_offsets[segment_sizes.size() - 1] + segment_sizes.back();

  thrust::device_vector<int> d_offsets = h_offsets;
  thrust::device_vector<bool> d_mask   = generate(h_offsets.size() - 1, entropy);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    filter_out_segments_rle_scan(d_values, d_offsets, d_mask);
  });
#endif
}

// Step 2: Implement using segment ids and cub::DeviceSelect::If
template <typename T>
static void filter_out_segments_copy_if(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
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
#else
  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  constexpr int max_segment_size = 20;
  const int max_num_segments     = cuda::ceil_div(elements, max_segment_size);

  thrust::device_vector<T> d_values = generate(elements);

  auto segment_sizes = partition_into_segments(elements, max_segment_size);
  thrust::host_vector<int> h_offsets(segment_sizes.size() + 1);
  std::exclusive_scan(segment_sizes.begin(), segment_sizes.end(), h_offsets.begin(), 0);
  h_offsets[segment_sizes.size()] = h_offsets[segment_sizes.size() - 1] + segment_sizes.back();

  thrust::device_vector<int> d_offsets = h_offsets;
  thrust::device_vector<bool> d_mask   = generate(h_offsets.size() - 1, entropy);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    filter_out_segments_copy_if(d_values, d_offsets, d_mask);
  });
#endif
}

// Step 3: Implement using fancy iterator to get segment ids on-the-fly
template <typename T>
static void filter_out_segments_fancy_iterator(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
  // Example: [[30], [40,20], [50], [10,30,80]]
  // Flatten:
  thrust::device_vector<T> d_values{30, 40, 20, 50, 10, 30, 80};
  thrust::device_vector<int> d_offsets{0, 1, 3, 4, 7}; // 4 segments
  thrust::device_vector<bool> d_mask{true, false, false, true}; // Keep segments 0 and 3
  std::cout << "Before filtering:" << std::endl;
  print_array(d_values, d_offsets);

  filter_out_segments_fancy_iterator(d_values, d_offsets, d_mask);

  std::cout << "After filtering:" << std::endl;
  print_array(d_values, d_offsets);
#else
  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  constexpr int max_segment_size = 20;
  const int max_num_segments     = cuda::ceil_div(elements, max_segment_size);

  thrust::device_vector<T> d_values = generate(elements);

  auto segment_sizes = partition_into_segments(elements, max_segment_size);
  thrust::host_vector<int> h_offsets(segment_sizes.size() + 1);
  std::exclusive_scan(segment_sizes.begin(), segment_sizes.end(), h_offsets.begin(), 0);
  h_offsets[segment_sizes.size()] = h_offsets[segment_sizes.size() - 1] + segment_sizes.back();

  thrust::device_vector<int> d_offsets = h_offsets;
  thrust::device_vector<bool> d_mask   = generate(h_offsets.size() - 1, entropy);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    filter_out_segments_fancy_iterator(d_values, d_offsets, d_mask);
  });
#endif
}

using current_data_types = nvbench::type_list<float>;

NVBENCH_BENCH_TYPES(filter_out_segments_rle_scan, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("filter_out_segments_rle_scan")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 24, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});

NVBENCH_BENCH_TYPES(filter_out_segments_copy_if, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("filter_out_segments_copy_if")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 24, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});

NVBENCH_BENCH_TYPES(filter_out_segments_fancy_iterator, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("filter_out_segments_fancy_iterator")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 24, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
