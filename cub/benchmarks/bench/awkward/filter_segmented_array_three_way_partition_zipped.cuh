#pragma once

#include <cub/device/device_partition.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>

template <typename T, typename PredicateOp>
struct partition_keep_zipped_op
{
  PredicateOp pred;
  const T* pt;
  const T* eta;
  const T* phi;

  __device__ bool operator()(int global_index) const
  {
    return pred(cuda::std::tuple<T, T, T>{pt[global_index], eta[global_index], phi[global_index]});
  }
};

template <typename T>
struct write_selected_zipped_from_index_op
{
  const T* pt;
  const T* eta;
  const T* phi;

  __device__ cuda::std::tuple<T, T, T> operator()(int global_index) const
  {
    return cuda::std::tuple<T, T, T>{pt[global_index], eta[global_index], phi[global_index]};
  }
};

struct record_rejected_segment_from_index_op
{
  const int* offsets;
  int num_segments;
  int* num_removed_per_segment;

  __device__ int operator()(int global_index) const
  {
    const int* it           = thrust::lower_bound(thrust::seq, offsets, offsets + num_segments + 1, global_index + 1);
    const int segment_index = static_cast<int>(it - offsets - 1);
    atomicAdd(num_removed_per_segment + segment_index, 1);
    return global_index;
  }
};

struct always_second_partition_op
{
  __device__ bool operator()(int) const
  {
    return true;
  }
};

struct adjust_offsets_three_way_partition_zipped_op
{
  const int* num_removed_per_segment;
  int num_segments;
  const int* input_offsets;

  __device__ int operator()(int segment_index) const
  {
    if (segment_index == num_segments)
    {
      return 0;
    }

    const int segment_start = input_offsets[segment_index];
    const int segment_end   = input_offsets[segment_index + 1];
    return (segment_end - segment_start) - num_removed_per_segment[segment_index];
  }
};

template <typename T, typename PredicateOp>
static void segmented_filter_three_way_partition_zipped(
  const thrust::device_vector<T>& d_pt,
  const thrust::device_vector<T>& d_eta,
  const thrust::device_vector<T>& d_phi,
  const thrust::device_vector<int>& d_offsets,
  thrust::device_vector<T>& d_selected_pt,
  thrust::device_vector<T>& d_selected_eta,
  thrust::device_vector<T>& d_selected_phi,
  thrust::device_vector<int>& d_new_offsets,
  thrust::device_vector<int>& d_num_selected_out,
  thrust::device_vector<int>& d_num_removed_per_segment,
  thrust::device_vector<uint8_t>& d_temp_storage,
  PredicateOp pred)
{
  const auto num_segments = d_offsets.size() - 1;

  d_selected_pt.resize(d_pt.size());
  d_selected_eta.resize(d_eta.size());
  d_selected_phi.resize(d_phi.size());
  d_new_offsets.resize(num_segments + 1);
  d_num_selected_out.resize(2);
  d_num_removed_per_segment.assign(num_segments, 0);

  auto selected_output_iter = cuda::make_transform_output_iterator(
    cuda::make_zip_iterator(thrust::raw_pointer_cast(d_selected_pt.data()),
                            thrust::raw_pointer_cast(d_selected_eta.data()),
                            thrust::raw_pointer_cast(d_selected_phi.data())),
    write_selected_zipped_from_index_op<T>{
      thrust::raw_pointer_cast(d_pt.data()),
      thrust::raw_pointer_cast(d_eta.data()),
      thrust::raw_pointer_cast(d_phi.data())});

  auto rejected_output_iter = cuda::make_transform_output_iterator(
    thrust::make_discard_iterator(),
    record_rejected_segment_from_index_op{
      thrust::raw_pointer_cast(d_offsets.data()),
      static_cast<int>(num_segments),
      thrust::raw_pointer_cast(d_num_removed_per_segment.data())});

  partition_keep_zipped_op<T, PredicateOp> select_first_part_op{
    pred,
    thrust::raw_pointer_cast(d_pt.data()),
    thrust::raw_pointer_cast(d_eta.data()),
    thrust::raw_pointer_cast(d_phi.data())};

  size_t temp_storage_bytes = 0;
  auto error                = cub::DevicePartition::If(
    nullptr,
    temp_storage_bytes,
    cuda::counting_iterator{0},
    selected_output_iter,
    rejected_output_iter,
    thrust::make_discard_iterator(),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_first_part_op,
    always_second_partition_op{});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temp storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  if (d_temp_storage.size() < temp_storage_bytes)
  {
    d_temp_storage.resize(temp_storage_bytes);
  }

  error = cub::DevicePartition::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::counting_iterator{0},
    selected_output_iter,
    rejected_output_iter,
    thrust::make_discard_iterator(),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_first_part_op,
    always_second_partition_op{});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during DevicePartition::If: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  const int num_selected = d_num_selected_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  adjust_offsets_three_way_partition_zipped_op adjust_op{
    thrust::raw_pointer_cast(d_num_removed_per_segment.data()),
    static_cast<int>(num_segments),
    thrust::raw_pointer_cast(d_offsets.data())};

  auto input_iter = cuda::make_transform_iterator(cuda::counting_iterator{0}, adjust_op);

  error = cub::DeviceScan::ExclusiveScan(
    nullptr,
    temp_storage_bytes,
    input_iter,
    thrust::raw_pointer_cast(d_new_offsets.data()),
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
    input_iter,
    thrust::raw_pointer_cast(d_new_offsets.data()),
    cuda::std::plus<>{},
    0,
    num_segments + 1);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during ExclusiveScan: " << cudaGetErrorString(error) << std::endl;
  }
}
