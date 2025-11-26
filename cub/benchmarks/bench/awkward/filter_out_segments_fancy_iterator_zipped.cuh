#pragma once

#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

template <typename T>
static void filter_out_segments_fancy_iterator_zipped(
  thrust::device_vector<T>& d_pt,
  thrust::device_vector<T>& d_eta,
  thrust::device_vector<T>& d_phi,
  thrust::device_vector<int>& d_offsets,
  thrust::device_vector<T>& d_selected_pt,
  thrust::device_vector<T>& d_selected_eta,
  thrust::device_vector<T>& d_selected_phi,
  thrust::device_vector<int>& d_selected_segment_ids,
  thrust::device_vector<int>& d_num_selected_out,
  thrust::device_vector<uint8_t>& d_temp_storage,
  const thrust::device_vector<bool>& d_mask)
{
  auto select_op =
    [d_mask = thrust::raw_pointer_cast(d_mask.data())] __device__(const cuda::std::tuple<T, T, T, int>& t) -> bool {
    int segment_id = cuda::std::get<3>(t);
    return d_mask[segment_id];
  };

  int num_segments = static_cast<int>(d_offsets.size() - 1);

  auto fancy_iterator = cuda::make_transform_iterator(
    cuda::counting_iterator{0},
    [offsets = thrust::raw_pointer_cast(d_offsets.data()), num_segments] __device__(int global_index) -> int {
      // Determine which segment this index belongs to
      const int* it     = thrust::lower_bound(thrust::seq, offsets, offsets + num_segments + 1, global_index + 1);
      int segment_index = static_cast<int>(it - offsets - 1);
      return segment_index;
    });

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::make_zip_iterator(d_pt.begin(), d_eta.begin(), d_phi.begin(), fancy_iterator),
    cuda::make_zip_iterator(
      d_selected_pt.begin(), d_selected_eta.begin(), d_selected_phi.begin(), d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  // thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::make_zip_iterator(d_pt.begin(), d_eta.begin(), d_phi.begin(), fancy_iterator),
    cuda::make_zip_iterator(
      d_selected_pt.begin(), d_selected_eta.begin(), d_selected_phi.begin(), d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  thrust::device_vector<int> d_num_segments_out(1, thrust::no_init);

  int num_selected = d_num_selected_out[0];

  auto copy_boundaries_op =
    [num_selected, d_selected_segment_ids = thrust::raw_pointer_cast(d_selected_segment_ids.data())] __device__(
      int segment_id) -> bool {
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
    d_offsets.begin(),
    thrust::raw_pointer_cast(d_num_segments_out.data()),
    num_selected,
    copy_boundaries_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  // d_temp_storage.resize(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::counting_iterator{0},
    d_offsets.begin(),
    thrust::raw_pointer_cast(d_num_segments_out.data()),
    num_selected,
    copy_boundaries_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  int new_num_segments = d_num_segments_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  d_offsets.resize(new_num_segments + 1);
  d_offsets[new_num_segments] = num_selected;
}
