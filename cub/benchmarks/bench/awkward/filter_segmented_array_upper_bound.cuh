#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// Step 3: Use upper_bound to get segment ids
template <typename T, typename PredicateOp>
static void segmented_filter_upper_bound(
  thrust::device_vector<T>& d_values, thrust::device_vector<int>& d_offsets, PredicateOp pred)
{
  thrust::device_vector<int> d_segment_ids(d_values.size(), thrust::no_init);
  thrust::upper_bound(
    thrust::device,
    d_offsets.begin(),
    d_offsets.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(d_values.size())},
    cuda::make_transform_output_iterator(d_segment_ids.begin(), [] __device__(int value) {
      return value - 1;
    }));

  thrust::device_vector<T> d_selected_values(d_values.size(), thrust::no_init);
  thrust::device_vector<int> d_selected_segment_ids(d_values.size(), thrust::no_init);
  thrust::device_vector<int> d_num_selected_out(1, thrust::no_init);
  auto select_op = [pred] __device__(const cuda::std::tuple<T, int>& t) {
    T value = cuda::std::get<0>(t);
    return pred(value);
  };

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::make_zip_iterator(d_values.begin(), d_segment_ids.begin()),
    cuda::make_zip_iterator(d_selected_values.begin(), d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_values.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);
  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::make_zip_iterator(d_values.begin(), d_segment_ids.begin()),
    cuda::make_zip_iterator(d_selected_values.begin(), d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_values.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  int num_selected = d_num_selected_out[0];

  d_selected_values.resize(num_selected);
  d_values.swap(d_selected_values);
  d_selected_segment_ids.resize(num_selected);

  thrust::lower_bound(
    thrust::device,
    d_selected_segment_ids.begin(),
    d_selected_segment_ids.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(d_offsets.size())},
    d_offsets.begin());
}
