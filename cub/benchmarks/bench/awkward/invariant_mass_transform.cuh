#pragma once

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>

template <typename T>
static thrust::device_vector<T> invariant_mass_transform(
  const thrust::device_vector<T>& d_pt,
  const thrust::device_vector<T>& d_eta,
  const thrust::device_vector<T>& d_phi,
  int segment_size)
{
  // Right now this implementation is hardcoded to assume stride == 2.
  // We would need a zip iterator of `stride` iterators instead of
  // first and second electron iterators.
  auto first_electron_iter = cuda::make_zip_iterator(
    cuda::make_strided_iterator(d_pt.begin(), segment_size),
    cuda::make_strided_iterator(d_eta.begin(), segment_size),
    cuda::make_strided_iterator(d_phi.begin(), segment_size));

  auto second_electron_iter = cuda::make_zip_iterator(
    cuda::make_strided_iterator(d_pt.begin() + 1, segment_size),
    cuda::make_strided_iterator(d_eta.begin() + 1, segment_size),
    cuda::make_strided_iterator(d_phi.begin() + 1, segment_size));

  auto input_iter = cuda::std::make_tuple(first_electron_iter, second_electron_iter);

  const auto num_elements = d_pt.size() / 2;
  thrust::device_vector<T> d_output(num_elements, thrust::no_init);
  auto op =
    [] __device__(const cuda::std::tuple<T, T, T>& first_electron, const cuda::std::tuple<T, T, T>& second_electron) {
      auto [pt1, eta1, phi1] = first_electron;
      auto [pt2, eta2, phi2] = second_electron;

      auto m2 = 2 * pt1 * pt2 * (cuda::std::cosh(eta1 - eta2) - cuda::std::cos(phi1 - phi2));
      return cuda::std::sqrt(m2);
    };

  auto error = cub::DeviceTransform::Transform(input_iter, thrust::raw_pointer_cast(d_output.data()), num_elements, op);
  if (error != cudaSuccess)
  {
    std::cerr << "Error during segmented reduce: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  return d_output;
}
