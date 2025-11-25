#pragma once

#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <tuple>

template <typename T>
static std::
  tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<int>>
  filter_out_segments_copy_if_zipped(
    const thrust::device_vector<T>& d_pt,
    const thrust::device_vector<T>& d_eta,
    const thrust::device_vector<T>& d_phi,
    const thrust::device_vector<int>& d_offsets,
    const thrust::device_vector<bool>& d_mask)
{
  thrust::device_vector<int> d_new_offsets = d_offsets;
  thrust::device_vector<T> d_selected_pt(d_pt.size(), thrust::no_init);
  thrust::device_vector<T> d_selected_eta(d_eta.size(), thrust::no_init);
  thrust::device_vector<T> d_selected_phi(d_phi.size(), thrust::no_init);
  thrust::device_vector<int> d_selected_segment_ids(d_pt.size(), thrust::no_init);
  thrust::device_vector<int> d_num_selected_out(1, thrust::no_init);

  thrust::device_vector<int> d_segment_ids(d_pt.size(), thrust::no_init);
  thrust::upper_bound(
    thrust::device,
    d_new_offsets.begin(),
    d_new_offsets.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(d_pt.size())},
    cuda::make_transform_output_iterator(d_segment_ids.begin(), [] __device__(int value) {
      return value - 1;
    }));

  auto select_op =
    [d_mask = thrust::raw_pointer_cast(d_mask.data())] __device__(const cuda::std::tuple<T, T, T, int>& t) {
      int segment_id = cuda::std::get<3>(t);
      return d_mask[segment_id];
    };

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::make_zip_iterator(d_pt.begin(), d_eta.begin(), d_phi.begin(), d_segment_ids.begin()),
    cuda::make_zip_iterator(
      d_selected_pt.begin(), d_selected_eta.begin(), d_selected_phi.begin(), d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
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
    cuda::make_zip_iterator(d_pt.begin(), d_eta.begin(), d_phi.begin(), d_segment_ids.begin()),
    cuda::make_zip_iterator(
      d_selected_pt.begin(), d_selected_eta.begin(), d_selected_phi.begin(), d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  thrust::device_vector<int> d_num_segments_out(1, thrust::no_init);

  int num_selected = d_num_selected_out[0];

  auto copy_boundaries_op =
    [num_selected,
     d_selected_segment_ids = thrust::raw_pointer_cast(d_selected_segment_ids.data())] __device__(int segment_id) {
      if (segment_id == 0)
      {
        return true;
      }
      return d_selected_segment_ids[segment_id] != d_selected_segment_ids[segment_id - 1];
    };

  error = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::counting_iterator{0},
    d_new_offsets.begin(),
    thrust::raw_pointer_cast(d_num_segments_out.data()),
    num_selected,
    copy_boundaries_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  d_temp_storage.resize(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::counting_iterator{0},
    d_new_offsets.begin(),
    thrust::raw_pointer_cast(d_num_segments_out.data()),
    num_selected,
    copy_boundaries_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  int num_segments = d_num_segments_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  d_new_offsets.resize(num_segments + 1);
  d_new_offsets[num_segments] = num_selected;

  return {d_selected_pt, d_selected_eta, d_selected_phi, d_new_offsets};
}
