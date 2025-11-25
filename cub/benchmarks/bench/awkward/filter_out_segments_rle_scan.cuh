#pragma once

#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

template <typename T>
static std::tuple<thrust::device_vector<T>, thrust::device_vector<int>> filter_out_segments_rle_scan(
  const thrust::device_vector<T>& d_values,
  const thrust::device_vector<int>& d_offsets,
  const thrust::device_vector<bool>& d_mask)
{
  thrust::device_vector<int> d_new_offsets = d_offsets;
  thrust::device_vector<int> d_segment_ids(d_values.size(), thrust::no_init);
  thrust::upper_bound(
    thrust::device,
    d_new_offsets.begin(),
    d_new_offsets.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(d_values.size())},
    cuda::make_transform_output_iterator(d_segment_ids.begin(), [] __device__(int value) {
      return value - 1;
    }));

  thrust::device_vector<T> d_selected_values(d_values.size(), thrust::no_init);
  thrust::device_vector<int> d_selected_segment_ids(d_values.size(), thrust::no_init);
  thrust::device_vector<int> d_num_selected_out(1, thrust::no_init);
  auto select_op = [d_mask = thrust::raw_pointer_cast(d_mask.data())] __device__(const cuda::std::tuple<T, int>& t) {
    int segment_id = cuda::std::get<1>(t);
    return d_mask[segment_id];
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
    return {};
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
    return {};
  }

  thrust::device_vector<int> d_new_segment_sizes(d_offsets.size() - 1, thrust::no_init);
  thrust::device_vector<int> d_num_segments_out(1, thrust::no_init);

  int num_selected = d_num_selected_out[0];

  error = cub::DeviceRunLengthEncode::Encode(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_selected_segment_ids.data()),
    cuda::make_discard_iterator(),
    thrust::raw_pointer_cast(d_new_segment_sizes.data()),
    thrust::raw_pointer_cast(d_num_segments_out.data()),
    num_selected);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  d_temp_storage.resize(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceRunLengthEncode::Encode(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_selected_segment_ids.data()),
    cuda::make_discard_iterator(),
    thrust::raw_pointer_cast(d_new_segment_sizes.data()),
    thrust::raw_pointer_cast(d_num_segments_out.data()),
    num_selected);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during run-length encoding: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  int num_segments = d_num_segments_out[0];

  error = cub::DeviceScan::ExclusiveSum(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_new_segment_sizes.data()),
    thrust::raw_pointer_cast(d_new_offsets.data()),
    num_segments);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temp storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  d_temp_storage.resize(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceScan::ExclusiveSum(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_new_segment_sizes.data()),
    thrust::raw_pointer_cast(d_new_offsets.data()),
    num_segments);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during exclusive sum: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  d_selected_values.resize(num_selected);
  d_new_offsets.resize(num_segments + 1);
  d_new_offsets[num_segments] = num_selected;

  return {d_selected_values, d_new_offsets};
}
