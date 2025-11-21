#pragma once

// Step 4: Zip three input value arrays
template <typename T>
static void segmented_filter_upper_bound_zipped(
  thrust::device_vector<T>& d_pt,
  thrust::device_vector<T>& d_eta,
  thrust::device_vector<T>& d_phi,
  thrust::device_vector<int>& d_offsets,
  T threshold)
{
  auto num_values = d_pt.size();

  thrust::device_vector<int> d_segment_ids(num_values, thrust::no_init);
  thrust::upper_bound(
    thrust::device,
    d_offsets.begin(),
    d_offsets.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(num_values)},
    cuda::make_transform_output_iterator(d_segment_ids.begin(), [] __device__(int value) {
      return value - 1;
    }));

  // Create zipped input and output iterators
  thrust::device_vector<T> d_selected_pt(num_values, thrust::no_init);
  thrust::device_vector<T> d_selected_eta(num_values, thrust::no_init);
  thrust::device_vector<T> d_selected_phi(num_values, thrust::no_init);

  auto input_values_iter = cuda::make_zip_iterator(d_pt.begin(), d_eta.begin(), d_phi.begin());
  auto output_values_iter =
    cuda::make_zip_iterator(d_selected_pt.begin(), d_selected_eta.begin(), d_selected_phi.begin());

  thrust::device_vector<int> d_selected_segment_ids(num_values, thrust::no_init);
  thrust::device_vector<int> d_num_selected_out(1, thrust::no_init);

  // Selecting based on value from the pt array
  auto select_op = [threshold] __device__(const cuda::std::tuple<cuda::std::tuple<T, T, T>, int>& t) {
    T value = cuda::std::get<0>(cuda::std::get<0>(t));
    return value >= threshold;
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
    return;
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
    return;
  }

  int num_selected = d_num_selected_out[0];

  d_selected_pt.resize(num_selected);
  d_selected_eta.resize(num_selected);
  d_selected_phi.resize(num_selected);
  d_pt.swap(d_selected_pt);
  d_eta.swap(d_selected_eta);
  d_phi.swap(d_selected_phi);

  d_selected_segment_ids.resize(num_selected);

  thrust::lower_bound(
    thrust::device,
    d_selected_segment_ids.begin(),
    d_selected_segment_ids.end(),
    cuda::counting_iterator{0},
    cuda::counting_iterator{static_cast<int>(d_offsets.size())},
    d_offsets.begin());
}
