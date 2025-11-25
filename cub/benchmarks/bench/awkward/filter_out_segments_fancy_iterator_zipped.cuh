#pragma once

#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

template <typename T>
struct fancy_select_mask_op
{
  const bool* mask;

  __host__ __device__ bool operator()(const cuda::std::tuple<T, T, T, int>& t) const
  {
    const int segment_id = cuda::std::get<3>(t);
    return mask[segment_id];
  }
};

struct fancy_segment_lookup_op
{
  const int* offsets;
  int num_segments;

  __host__ __device__ int operator()(int global_index) const
  {
    const int* it = thrust::lower_bound(thrust::seq, offsets, offsets + num_segments + 1, global_index + 1);
    return static_cast<int>(it - offsets - 1);
  }
};

struct fancy_copy_boundaries_op
{
  const int* selected_segment_ids;

  __host__ __device__ bool operator()(int segment_id) const
  {
    if (segment_id == 0)
    {
      return true;
    }
    return selected_segment_ids[segment_id] != selected_segment_ids[segment_id - 1];
  }
};

template <typename T>
static std::
  tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<int>>
  filter_out_segments_fancy_iterator_zipped(
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
  fancy_select_mask_op<T> select_op{thrust::raw_pointer_cast(d_mask.data())};

  int num_segments = static_cast<int>(d_offsets.size() - 1);
  fancy_segment_lookup_op segment_lookup{thrust::raw_pointer_cast(d_new_offsets.data()), num_segments};
  auto fancy_iterator = cuda::make_transform_iterator(cuda::counting_iterator{0}, segment_lookup);

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
    return {};
  }

  thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

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
    return {};
  }

  thrust::device_vector<int> d_num_segments_out(1, thrust::no_init);

  int num_selected = d_num_selected_out[0];

  fancy_copy_boundaries_op copy_boundaries_op{thrust::raw_pointer_cast(d_selected_segment_ids.data())};

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

  int new_num_segments = d_num_segments_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  d_new_offsets.resize(new_num_segments + 1);
  d_new_offsets[new_num_segments] = num_selected;

  return {d_selected_pt, d_selected_eta, d_selected_phi, d_new_offsets};
}
