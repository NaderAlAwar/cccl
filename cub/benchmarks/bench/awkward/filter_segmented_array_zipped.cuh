#pragma once

#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h> // thrust::lower_bound
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <nvbench_helper.cuh>

template <typename T, typename PredicateOp>
struct stateful_select_zipped_op
{
  PredicateOp pred;
  const T* pt;
  const T* eta;
  const T* phi;
  const int* offsets;
  int num_segments;
  int* num_removed_per_segment;

  __device__ bool operator()(int global_index) const
  {
    bool keep = pred(cuda::std::tuple<T, T, T>{pt[global_index], eta[global_index], phi[global_index]});

    if (!keep)
    {
      // figure out which segment this value belongs to
      // increment num_removed_per_segment for that segment

      // `it` is a memory location in the offsets array, so we still need to
      // find the segment index. We do this with pointer arithmetic.
      const int* it = thrust::lower_bound(thrust::seq, offsets, offsets + num_segments + 1, global_index + 1);

      int segment_index = static_cast<int>(it - offsets - 1);

      atomicAdd(&(num_removed_per_segment[segment_index]), 1);
    }

    return keep;
  }
};

struct adjust_offsets_zipped_op
{
  const int* num_removed_per_segment;
  int num_segments;
  const int* input_offsets; // size num_segments + 1

  __device__ int operator()(int segment_index) const
  {
    if (segment_index == num_segments)
    {
      return 0;
    }

    int segment_start = input_offsets[segment_index];
    int segment_end   = input_offsets[segment_index + 1];

    int new_length = (segment_end - segment_start) - num_removed_per_segment[segment_index];

    return new_length;
  }
};

template <typename T, typename PredicateOp>
static void segmented_filter_zipped(
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

  // Write directly to d_values
  auto output_iter = cuda::make_transform_output_iterator(
    cuda::make_zip_iterator(thrust::raw_pointer_cast(d_selected_pt.data()),
                            thrust::raw_pointer_cast(d_selected_eta.data()),
                            thrust::raw_pointer_cast(d_selected_phi.data())),
    [pt  = thrust::raw_pointer_cast(d_pt.data()),
     eta = thrust::raw_pointer_cast(d_eta.data()),
     phi = thrust::raw_pointer_cast(d_phi.data())] __device__(int idx) {
      return cuda::std::make_tuple(pt[idx], eta[idx], phi[idx]);
    });

  stateful_select_zipped_op<T, PredicateOp> select_op{
    pred,
    thrust::raw_pointer_cast(d_pt.data()),
    thrust::raw_pointer_cast(d_eta.data()),
    thrust::raw_pointer_cast(d_phi.data()),
    thrust::raw_pointer_cast(d_offsets.data()),
    static_cast<int>(num_segments),
    thrust::raw_pointer_cast(d_num_removed_per_segment.data())};

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::counting_iterator{0},
    output_iter,
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temp storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  // thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::counting_iterator{0},
    output_iter,
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_pt.size(),
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during DeviceSelect::If: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  int num_selected = thrust::host_vector<int>(d_num_selected_out)[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  adjust_offsets_zipped_op adjust_op{
    thrust::raw_pointer_cast(d_num_removed_per_segment.data()),
    static_cast<int>(num_segments),
    thrust::raw_pointer_cast(d_offsets.data())};

  auto input_iter = cuda::make_transform_iterator(cuda::counting_iterator{0}, adjust_op);
  // thrust::device_vector<int> d_new_offsets(num_segments + 1, thrust::no_init);

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

  // d_temp_storage.resize(temp_storage_bytes);

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
    return;
  }
}
