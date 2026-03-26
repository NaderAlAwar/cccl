#pragma once

#include <cub/device/device_select.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

template <typename T, typename PredicateOp>
static void segmented_filter_lower_bound_only_zipped(
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

  auto num_values = d_pt.size();

  thrust::device_vector<int> d_selected_indices(d_pt.size(), thrust::no_init);

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
    cuda::make_zip_iterator(input_values_iter, cuda::counting_iterator<int>{0}),
    cuda::make_zip_iterator(output_values_iter, d_selected_indices.begin()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    num_values,
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
    cuda::make_zip_iterator(input_values_iter, cuda::counting_iterator<int>{0}),
    cuda::make_zip_iterator(output_values_iter, d_selected_indices.begin()),
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
  d_selected_indices.resize(num_selected);

  // For each segment boundary, compute how many selected indices are < offset
  thrust::lower_bound(
    thrust::device,
    d_selected_indices.begin(),
    d_selected_indices.end(),
    d_offsets.begin(),
    d_offsets.end(),
    d_new_offsets.begin());
}
