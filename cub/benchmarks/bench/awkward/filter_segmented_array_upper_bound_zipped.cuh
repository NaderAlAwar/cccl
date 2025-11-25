#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

template <typename T, typename PredicateOp>
static std::
  tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<int>>
  segmented_filter_upper_bound_zipped(
    const thrust::device_vector<T>& d_pt,
    const thrust::device_vector<T>& d_eta,
    const thrust::device_vector<T>& d_phi,
    const thrust::device_vector<int>& d_offsets,
    PredicateOp pred)
{
  thrust::device_vector<int> d_new_offsets = d_offsets;
  thrust::device_vector<T> d_selected_pt(d_pt.size(), thrust::no_init);
  thrust::device_vector<T> d_selected_eta(d_eta.size(), thrust::no_init);
  thrust::device_vector<T> d_selected_phi(d_phi.size(), thrust::no_init);
  thrust::device_vector<int> d_selected_segment_ids(d_pt.size(), thrust::no_init);
  thrust::device_vector<int> d_num_selected_out(1, thrust::no_init);

  auto num_values = d_pt.size();

  thrust::device_vector<int> d_segment_ids(num_values, thrust::no_init);
  thrust::upper_bound(
    thrust::device,
    d_new_offsets.begin(),
    d_new_offsets.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(num_values)},
    cuda::make_transform_output_iterator(d_segment_ids.begin(), [] __device__(int value) {
      return value - 1;
    }));

  // Create zipped input and output iterators
  auto input_values_iter = cuda::make_zip_iterator(
    thrust::raw_pointer_cast(d_pt.data()),
    thrust::raw_pointer_cast(d_eta.data()),
    thrust::raw_pointer_cast(d_phi.data()));
  auto output_values_iter =
    cuda::make_zip_iterator(d_selected_pt.begin(), d_selected_eta.begin(), d_selected_phi.begin());

  // Selecting based on predicate that receives a tuple of (pt, eta, phi)
  auto select_op = [pred] __device__(const cuda::std::tuple<cuda::std::tuple<T, T, T>, int>& t) {
    const auto& values = cuda::std::get<0>(t);
    return pred(values);
  };

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    cuda::make_zip_iterator(input_values_iter, d_segment_ids.begin()),
    cuda::make_zip_iterator(output_values_iter, d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    num_values,
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
    cuda::make_zip_iterator(input_values_iter, d_segment_ids.begin()),
    cuda::make_zip_iterator(output_values_iter, d_selected_segment_ids.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    num_values,
    select_op);

  if (error != cudaSuccess)
  {
    std::cerr << "Error during selection: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  int num_selected = d_num_selected_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);

  d_selected_segment_ids.resize(num_selected);

  thrust::lower_bound(
    thrust::device,
    d_selected_segment_ids.begin(),
    d_selected_segment_ids.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(d_new_offsets.size())},
    d_new_offsets.begin());

  return {d_selected_pt, d_selected_eta, d_selected_phi, d_new_offsets};
}
