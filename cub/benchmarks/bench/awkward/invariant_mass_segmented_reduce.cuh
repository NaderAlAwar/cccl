#pragma once

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>

template <typename T>
static thrust::device_vector<T> invariant_mass_segmented_reduce(
  const thrust::device_vector<T>& d_pt,
  const thrust::device_vector<T>& d_eta,
  const thrust::device_vector<T>& d_phi,
  int segment_size)
{
  auto input_iter         = cuda::make_zip_iterator(d_pt.begin(), d_eta.begin(), d_phi.begin());
  const auto num_elements = d_pt.size() / segment_size;
  auto op                 = [] __host__ __device__(const cuda::std::tuple<T, T, T>& first_electron,
                                                   const cuda::std::tuple<T, T, T>& second_electron) {
    auto [pt1, eta1, phi1] = first_electron;
    auto [pt2, eta2, phi2] = second_electron;

    return cuda::std::tuple{pt1 * pt2, eta1 - eta2, phi1 - phi2};
  };

  thrust::device_vector<T> d_output(num_elements, thrust::no_init);
  auto output_iter = cuda::make_transform_output_iterator(
    thrust::raw_pointer_cast(d_output.data()), [] __device__(const cuda::std::tuple<T, T, T>& t) {
      auto [pt, eta, phi] = t;
      return cuda::std::sqrt(2 * pt * (cuda::std::cosh(eta) - cuda::std::cos(phi)));
    });

  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceSegmentedReduce::Reduce(
    nullptr, temp_storage_bytes, input_iter, output_iter, num_elements, segment_size, op, cuda::std::tuple{1, 0, 0});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during temporary storage size calculation: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

  error = cub::DeviceSegmentedReduce::Reduce(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    input_iter,
    output_iter,
    num_elements,
    segment_size,
    op,
    cuda::std::tuple{1, 0, 0});

  if (error != cudaSuccess)
  {
    std::cerr << "Error during segmented reduce: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  return d_output;
}
