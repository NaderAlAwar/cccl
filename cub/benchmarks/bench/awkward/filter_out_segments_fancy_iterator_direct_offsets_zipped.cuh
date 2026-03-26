#pragma once

#include <cub/device/device_for.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda/functional>
#include <cuda/iterator>

#include <tuple>

struct kept_segment_length_op
{
  const int* offsets;
  const bool* mask;
  int num_segments;

  __device__ int operator()(int segment_index) const
  {
    if (segment_index == num_segments)
    {
      return 0;
    }

    return mask[segment_index] ? (offsets[segment_index + 1] - offsets[segment_index]) : 0;
  }
};

template <typename T>
struct scatter_kept_segments_zipped_op
{
  const T* pt;
  const T* eta;
  const T* phi;
  const int* offsets;
  const int* output_starts;
  const bool* mask;
  int num_segments;

  T* selected_pt;
  T* selected_eta;
  T* selected_phi;

  __device__ void operator()(int global_index) const
  {
    const int* it           = thrust::lower_bound(thrust::seq, offsets, offsets + num_segments + 1, global_index + 1);
    const int segment_index = static_cast<int>(it - offsets - 1);

    if (!mask[segment_index])
    {
      return;
    }

    const int out_idx     = output_starts[segment_index] + (global_index - offsets[segment_index]);
    selected_pt[out_idx]  = pt[global_index];
    selected_eta[out_idx] = eta[global_index];
    selected_phi[out_idx] = phi[global_index];
  }
};

struct keep_segment_boundary_op
{
  const bool* mask;
  int num_segments;

  __device__ bool operator()(int segment_index) const
  {
    return segment_index == num_segments || mask[segment_index];
  }
};

struct offset_from_segment_index_op
{
  const int* output_starts;

  __device__ int operator()(int segment_index) const
  {
    return output_starts[segment_index];
  }
};

template <typename T>
static void filter_out_segments_fancy_iterator_direct_offsets_zipped(
  const thrust::device_vector<T>& d_pt,
  const thrust::device_vector<T>& d_eta,
  const thrust::device_vector<T>& d_phi,
  thrust::device_vector<int>& d_offsets,
  thrust::device_vector<T>& d_selected_pt,
  thrust::device_vector<T>& d_selected_eta,
  thrust::device_vector<T>& d_selected_phi,
  thrust::device_vector<int>& d_selected_segment_ids,
  thrust::device_vector<int>& d_num_selected_out,
  thrust::device_vector<uint8_t>& d_temp_storage,
  const thrust::device_vector<bool>& d_mask)
{
  (void) d_selected_segment_ids; // unused in this implementation

  const int num_segments = static_cast<int>(d_offsets.size() - 1);

  thrust::device_vector<int> d_output_starts(num_segments + 1, thrust::no_init);
  d_num_selected_out.resize(1);

  auto kept_lengths_iter = cuda::make_transform_iterator(
    cuda::counting_iterator{0},
    kept_segment_length_op{
      thrust::raw_pointer_cast(d_offsets.data()), thrust::raw_pointer_cast(d_mask.data()), num_segments});

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceScan::ExclusiveScan(
    nullptr,
    temp_storage_bytes,
    kept_lengths_iter,
    thrust::raw_pointer_cast(d_output_starts.data()),
    cuda::std::plus<>{},
    0,
    num_segments + 1);

  if (error != cudaSuccess)
  {
    std::cerr
      << "Error during temp storage size calculation for ExclusiveScan: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  if (d_temp_storage.size() < temp_storage_bytes)
  {
    d_temp_storage.resize(temp_storage_bytes);
  }

  error = cub::DeviceScan::ExclusiveScan(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    kept_lengths_iter,
    thrust::raw_pointer_cast(d_output_starts.data()),
    cuda::std::plus<>{},
    0,
    num_segments + 1);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during ExclusiveScan: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  const int num_selected = d_output_starts[num_segments];
  d_num_selected_out[0]  = num_selected;

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  temp_storage_bytes = 0;
  error              = cub::DeviceFor::Bulk(
    nullptr,
    temp_storage_bytes,
    static_cast<int>(d_pt.size()),
    scatter_kept_segments_zipped_op<T>{
      thrust::raw_pointer_cast(d_pt.data()),
      thrust::raw_pointer_cast(d_eta.data()),
      thrust::raw_pointer_cast(d_phi.data()),
      thrust::raw_pointer_cast(d_offsets.data()),
      thrust::raw_pointer_cast(d_output_starts.data()),
      thrust::raw_pointer_cast(d_mask.data()),
      num_segments,
      thrust::raw_pointer_cast(d_selected_pt.data()),
      thrust::raw_pointer_cast(d_selected_eta.data()),
      thrust::raw_pointer_cast(d_selected_phi.data())});

  if (error != cudaSuccess)
  {
    std::cerr
      << "Error during temp storage size calculation for DeviceFor::Bulk: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  if (d_temp_storage.size() < temp_storage_bytes)
  {
    d_temp_storage.resize(temp_storage_bytes);
  }

  error = cub::DeviceFor::Bulk(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    static_cast<int>(d_pt.size()),
    scatter_kept_segments_zipped_op<T>{
      thrust::raw_pointer_cast(d_pt.data()),
      thrust::raw_pointer_cast(d_eta.data()),
      thrust::raw_pointer_cast(d_phi.data()),
      thrust::raw_pointer_cast(d_offsets.data()),
      thrust::raw_pointer_cast(d_output_starts.data()),
      thrust::raw_pointer_cast(d_mask.data()),
      num_segments,
      thrust::raw_pointer_cast(d_selected_pt.data()),
      thrust::raw_pointer_cast(d_selected_eta.data()),
      thrust::raw_pointer_cast(d_selected_phi.data())});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during DeviceFor::Bulk: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  thrust::device_vector<int> d_num_offsets_out(1, thrust::no_init);
  d_offsets.resize(num_segments + 1);

  auto compact_offsets_out = cuda::make_transform_output_iterator(
    d_offsets.begin(), offset_from_segment_index_op{thrust::raw_pointer_cast(d_output_starts.data())});

  temp_storage_bytes = 0;
  error              = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::counting_iterator{0},
    compact_offsets_out,
    thrust::raw_pointer_cast(d_num_offsets_out.data()),
    num_segments + 1,
    keep_segment_boundary_op{thrust::raw_pointer_cast(d_mask.data()), num_segments});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temp storage size calculation for compacting offsets: " << cudaGetErrorString(error)
              << std::endl;
    return;
  }

  if (d_temp_storage.size() < temp_storage_bytes)
  {
    d_temp_storage.resize(temp_storage_bytes);
  }

  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::counting_iterator{0},
    compact_offsets_out,
    thrust::raw_pointer_cast(d_num_offsets_out.data()),
    num_segments + 1,
    keep_segment_boundary_op{thrust::raw_pointer_cast(d_mask.data()), num_segments});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during compacting offsets: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  d_offsets.resize(d_num_offsets_out[0]);
}

template <typename T>
static std::
  tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<int>>
  filter_out_segments_fancy_iterator_direct_offsets_zipped(
    const thrust::device_vector<T>& d_pt,
    const thrust::device_vector<T>& d_eta,
    const thrust::device_vector<T>& d_phi,
    const thrust::device_vector<int>& d_offsets,
    const thrust::device_vector<bool>& d_mask)
{
  thrust::device_vector<T> d_selected_pt;
  thrust::device_vector<T> d_selected_eta;
  thrust::device_vector<T> d_selected_phi;
  thrust::device_vector<int> d_new_offsets = d_offsets;
  thrust::device_vector<int> d_selected_segment_ids;
  thrust::device_vector<int> d_num_selected_out;
  thrust::device_vector<uint8_t> d_temp_storage;

  filter_out_segments_fancy_iterator_direct_offsets_zipped(
    d_pt,
    d_eta,
    d_phi,
    d_new_offsets,
    d_selected_pt,
    d_selected_eta,
    d_selected_phi,
    d_selected_segment_ids,
    d_num_selected_out,
    d_temp_storage,
    d_mask);

  return {std::move(d_selected_pt), std::move(d_selected_eta), std::move(d_selected_phi), std::move(d_new_offsets)};
}
