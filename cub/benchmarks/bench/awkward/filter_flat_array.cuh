#pragma once

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <nvbench_helper.cuh>

// Step 1: Filter out a single array
template <typename T, typename PredicateOp>
static void filter(thrust::device_vector<T>& d_values, PredicateOp pred)
{
  thrust::device_vector<int> d_num_selected_out(1, thrust::no_init);

  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::If(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_values.data()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_values.size(),
    pred);

  thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

  cub::DeviceSelect::If(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(d_values.data()),
    thrust::raw_pointer_cast(d_num_selected_out.data()),
    d_values.size(),
    pred);

  // Retrieve number of selected elements
  thrust::host_vector<int> h_num_selected_out = d_num_selected_out;
  int num_selected                            = h_num_selected_out[0];
  d_values.resize(num_selected);
}
