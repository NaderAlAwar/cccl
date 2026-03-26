#pragma once

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>

template <typename T>
struct write_selected_zipped_segment_ends_op
{
  T* selected_pt;
  T* selected_eta;
  T* selected_phi;
  int* new_offsets;

  __device__ void operator()(::cuda::std::ptrdiff_t selected_index,
                             const cuda::std::tuple<cuda::std::tuple<T, T, T>, int>& x) const
  {
    const auto& values = cuda::std::get<0>(x);
    const int segment  = cuda::std::get<1>(x);
    const int out_idx  = static_cast<int>(selected_index);

    selected_pt[out_idx]  = cuda::std::get<0>(values);
    selected_eta[out_idx] = cuda::std::get<1>(values);
    selected_phi[out_idx] = cuda::std::get<2>(values);

    atomicMax(new_offsets + segment + 1, out_idx + 1);
  }
};

template <typename T, typename PredicateOp>
static void segmented_filter_upper_bound_fancy_iterator_zipped(
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
  (void) d_num_removed_per_segment; // unused in this implementation

  d_selected_pt.resize(d_pt.size());
  d_selected_eta.resize(d_eta.size());
  d_selected_phi.resize(d_phi.size());
  d_new_offsets.assign(d_offsets.size(), 0);
  d_num_selected_out.resize(1);

  const auto num_values  = d_pt.size();
  const int num_segments = static_cast<int>(d_offsets.size() - 1);
  const int* offsets     = thrust::raw_pointer_cast(d_offsets.data());
  auto input_values_iter = cuda::make_zip_iterator(
    thrust::raw_pointer_cast(d_pt.data()),
    thrust::raw_pointer_cast(d_eta.data()),
    thrust::raw_pointer_cast(d_phi.data()));
  auto fancy_segment_ids = cuda::make_transform_iterator(
    cuda::counting_iterator{0}, [offsets, num_segments] __device__(int global_index) -> int {
      return static_cast<int>(
        ::cuda::std::upper_bound(offsets, offsets + num_segments + 1, global_index) - offsets - 1);
    });

  auto select_op = [pred] __device__(const cuda::std::tuple<cuda::std::tuple<T, T, T>, int>& t) -> bool {
    const auto& values = cuda::std::get<0>(t);
    return pred(values);
  };

  auto output_iter = ::cuda::make_tabulate_output_iterator(write_selected_zipped_segment_ends_op<T>{
    thrust::raw_pointer_cast(d_selected_pt.data()),
    thrust::raw_pointer_cast(d_selected_eta.data()),
    thrust::raw_pointer_cast(d_selected_phi.data()),
    thrust::raw_pointer_cast(d_new_offsets.data())});

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::make_zip_iterator(input_values_iter, fancy_segment_ids),
    output_iter,
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    num_values,
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  if (d_temp_storage.size() < temp_storage_bytes)
  {
    d_temp_storage.resize(temp_storage_bytes);
  }

  error = cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    cuda::make_zip_iterator(input_values_iter, fancy_segment_ids),
    output_iter,
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    num_values,
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  int num_selected = d_num_selected_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  temp_storage_bytes = 0;
  error              = cub::DeviceScan::InclusiveScan(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_new_offsets.data()),
    thrust::raw_pointer_cast(d_new_offsets.data()),
    ::cuda::maximum<>{},
    d_new_offsets.size());

  if (error != cudaSuccess)
  {
    std::cerr
      << "Error during temp storage size calculation for InclusiveScan: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  if (d_temp_storage.size() < temp_storage_bytes)
  {
    d_temp_storage.resize(temp_storage_bytes);
  }

  error = cub::DeviceScan::InclusiveScan(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_new_offsets.data()),
    thrust::raw_pointer_cast(d_new_offsets.data()),
    ::cuda::maximum<>{},
    d_new_offsets.size());

  if (error != cudaSuccess)
  {
    std::cerr << "Error during InclusiveScan: " << cudaGetErrorString(error) << std::endl;
  }
}
