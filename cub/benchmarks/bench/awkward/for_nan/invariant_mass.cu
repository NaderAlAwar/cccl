// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/tuple>

#include <nvbench_helper.cuh>

template <typename T>
static void invariant_mass_transform(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int segment_size = 2;

  const auto segments = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  const auto elements = segments * segment_size;

  thrust::device_vector<T> pt  = generate(elements, bit_entropy::_1_000, T{10}, T{100});
  thrust::device_vector<T> eta = generate(elements, bit_entropy::_1_000, T{-3}, T{3});
  thrust::device_vector<T> phi = generate(elements, bit_entropy::_1_000, T{0}, T{6.2831853071795864769});
  thrust::device_vector<T> output(segments);

  auto first = cuda::make_zip_iterator(
    cuda::make_strided_iterator(pt.begin(), segment_size),
    cuda::make_strided_iterator(eta.begin(), segment_size),
    cuda::make_strided_iterator(phi.begin(), segment_size));

  auto second = cuda::make_zip_iterator(
    cuda::make_strided_iterator(pt.begin() + 1, segment_size),
    cuda::make_strided_iterator(eta.begin() + 1, segment_size),
    cuda::make_strided_iterator(phi.begin() + 1, segment_size));

  auto input = cuda::std::make_tuple(first, second);
  auto op    = [] __device__(const cuda::std::tuple<T, T, T>& lhs, const cuda::std::tuple<T, T, T>& rhs) -> T {
    const auto [pt1, eta1, phi1] = lhs;
    const auto [pt2, eta2, phi2] = rhs;
    return cuda::std::sqrt(T{2} * pt1 * pt2 * (cuda::std::cosh(eta1 - eta2) - cuda::std::cos(phi1 - phi2)));
  };

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(3 * elements, "Size");
  state.add_global_memory_writes<T>(segments);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceTransform::Transform(
      input, thrust::raw_pointer_cast(output.data()), output.size(), op, launch.get_stream().get_stream());
  });
}

template <typename T>
static void invariant_mass_segmented_reduce(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int segment_size = 2;

  const auto segments = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  const auto elements = segments * segment_size;

  thrust::device_vector<T> pt  = generate(elements, bit_entropy::_1_000, T{10}, T{100});
  thrust::device_vector<T> eta = generate(elements, bit_entropy::_1_000, T{-3}, T{3});
  thrust::device_vector<T> phi = generate(elements, bit_entropy::_1_000, T{0}, T{6.2831853071795864769});
  thrust::device_vector<T> output(segments);

  auto input     = cuda::make_zip_iterator(pt.begin(), eta.begin(), phi.begin());
  auto reduce_op = [] __host__ __device__(const cuda::std::tuple<T, T, T>& lhs, const cuda::std::tuple<T, T, T>& rhs) {
    const auto [pt1, eta1, phi1] = lhs;
    const auto [pt2, eta2, phi2] = rhs;
    return cuda::std::tuple{pt1 * pt2, eta1 - eta2, phi1 - phi2};
  };

  auto output_iter = cuda::make_transform_output_iterator(
    thrust::raw_pointer_cast(output.data()), [] __device__(const cuda::std::tuple<T, T, T>& reduced) -> T {
      const auto [pt_product, eta_delta, phi_delta] = reduced;
      return cuda::std::sqrt(T{2} * pt_product * (cuda::std::cosh(eta_delta) - cuda::std::cos(phi_delta)));
    });

  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
    nullptr,
    temp_storage_bytes,
    input,
    output_iter,
    output.size(),
    segment_size,
    reduce_op,
    cuda::std::tuple{T{1}, T{0}, T{0}});

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(3 * elements, "Size");
  state.add_global_memory_writes<T>(segments);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceSegmentedReduce::Reduce(
      thrust::raw_pointer_cast(temp_storage.data()),
      temp_storage_bytes,
      input,
      output_iter,
      output.size(),
      segment_size,
      reduce_op,
      cuda::std::tuple{T{1}, T{0}, T{0}},
      launch.get_stream().get_stream());
  });
}

using value_types = nvbench::type_list<float, double>;

NVBENCH_BENCH_TYPES(invariant_mass_transform, NVBENCH_TYPE_AXES(value_types))
  .set_name("invariant_mass_transform")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));

NVBENCH_BENCH_TYPES(invariant_mass_segmented_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("invariant_mass_segmented_reduce")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 24, 4));
