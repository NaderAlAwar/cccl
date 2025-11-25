// This file implements the end to end benchmark needed by awkward. This includes three steps:
// 1) A segmented filter implementation
// 2) Filtering out segments based on length
// 3) Invariant mass calculation

#include <cuda/std/numeric>

#include <cmath>

#include <nvbench_helper.cuh>

#include "filter_out_segments_copy_if_zipped.cuh"
#include "filter_out_segments_fancy_iterator_zipped.cuh"
#include "filter_out_segments_rle_scan_zipped.cuh"
#include "filter_segmented_array_upper_bound_zipped.cuh"
#include "filter_segmented_array_zipped.cuh"
#include "invariant_mass_segmented_reduce.cuh"
#include "invariant_mass_transform.cuh"

template <typename T>
static std::tuple<thrust::device_vector<T>, thrust::device_vector<T>> physics_analysis(
  const thrust::device_vector<T>& d_electron_pts,
  const thrust::device_vector<T>& d_electron_etas,
  const thrust::device_vector<T>& d_electron_phis,
  const thrust::device_vector<T>& d_muon_pts,
  const thrust::device_vector<T>& d_muon_etas,
  const thrust::device_vector<T>& d_muon_phis,
  const thrust::device_vector<int>& d_electron_offsets,
  const thrust::device_vector<int>& d_muon_offsets)
{
  auto cond_electron = [] __device__(const cuda::std::tuple<T, T, T>& x) {
    return cuda::std::get<0>(x) > 40.0;
  };
  auto cond_muon = [] __device__(const cuda::std::tuple<T, T, T>& x) {
    return cuda::std::get<0>(x) > 20.0 && cuda::std::abs(cuda::std::get<1>(x)) < 2.4;
  };

  // The stateful op version seems to perform better. Uncommenting out the upper bound version yields the same result
  // with different performance.
  auto [d_selected_electron_pts, d_selected_electron_etas, d_selected_electron_phis, d_new_electron_offsets] =
    segmented_filter_zipped(d_electron_pts, d_electron_etas, d_electron_phis, d_electron_offsets, cond_electron);
  auto [d_selected_muon_pts, d_selected_muon_etas, d_selected_muon_phis, d_new_muon_offsets] =
    segmented_filter_zipped(d_muon_pts, d_muon_etas, d_muon_phis, d_muon_offsets, cond_muon);
  // auto [d_selected_electron_pts, d_selected_electron_etas, d_selected_electron_phis, d_new_electron_offsets] =
  // segmented_filter_upper_bound_zipped(d_electron_pts, d_electron_etas, d_electron_phis,
  // d_electron_offsets,cond_electron); auto [d_selected_muon_pts, d_selected_muon_etas, d_selected_muon_phis,
  // d_new_muon_offsets] = segmented_filter_upper_bound_zipped(d_muon_pts, d_muon_etas, d_muon_phis, d_muon_offsets,
  // cond_muon);

  const auto num_electron_segments = d_new_electron_offsets.size() - 1;
  const auto num_muon_segments     = d_new_muon_offsets.size() - 1;

  constexpr int segment_size = 2;

  thrust::device_vector<bool> d_electron_segment_mask(num_electron_segments, thrust::no_init);
  auto error = cub::DeviceTransform::Transform(
    cuda::std::tuple{d_new_electron_offsets.begin(), d_new_electron_offsets.begin() + 1},
    thrust::raw_pointer_cast(d_electron_segment_mask.data()),
    num_electron_segments,
    [segment_size] __device__(int start, int end) {
      return (end - start) == segment_size;
    });

  if (error != cudaSuccess)
  {
    std::cerr << "Error during electron segment length calculation: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  thrust::device_vector<bool> d_muon_segment_mask(num_muon_segments, thrust::no_init);
  error = cub::DeviceTransform::Transform(
    cuda::std::tuple{d_new_muon_offsets.begin(), d_new_muon_offsets.begin() + 1},
    thrust::raw_pointer_cast(d_muon_segment_mask.data()),
    num_muon_segments,
    [segment_size] __device__(int start, int end) {
      return (end - start) == segment_size;
    });

  if (error != cudaSuccess)
  {
    std::cerr << "Error during muon segment length calculation: " << cudaGetErrorString(error) << std::endl;
    return {};
  }

  // The fancy iterator version seems to perform better. Uncommenting out the copy_if and rle_scan versions yields the
  // same result with different performance.
  auto [d_final_electron_pts, d_final_electron_etas, d_final_electron_phis, d_final_electron_offsets] =
    filter_out_segments_fancy_iterator_zipped(
      d_selected_electron_pts,
      d_selected_electron_etas,
      d_selected_electron_phis,
      d_new_electron_offsets,
      d_electron_segment_mask);
  auto [d_final_muon_pts, d_final_muon_etas, d_final_muon_phis, d_final_muon_offsets] =
    filter_out_segments_fancy_iterator_zipped(
      d_selected_muon_pts, d_selected_muon_etas, d_selected_muon_phis, d_new_muon_offsets, d_muon_segment_mask);
  // auto [d_final_electron_pts, d_final_electron_etas, d_final_electron_phis, d_final_electron_offsets] =
  // filter_out_segments_copy_if_zipped(d_selected_electron_pts, d_selected_electron_etas, d_selected_electron_phis,
  // d_new_electron_offsets, d_electron_segment_mask); auto [d_final_muon_pts, d_final_muon_etas, d_final_muon_phis,
  // d_final_muon_offsets] = filter_out_segments_copy_if_zipped(d_selected_muon_pts, d_selected_muon_etas,
  // d_selected_muon_phis, d_new_muon_offsets, d_muon_segment_mask); auto [d_final_electron_pts, d_final_electron_etas,
  // d_final_electron_phis, d_final_electron_offsets] = filter_out_segments_copy_if_zipped(d_selected_electron_pts,
  // d_selected_electron_etas, d_selected_electron_phis, d_new_electron_offsets, d_electron_segment_mask); auto
  // [d_final_muon_pts, d_final_muon_etas, d_final_muon_phis, d_final_muon_offsets] =
  // filter_out_segments_rle_scan_zipped(d_selected_muon_pts, d_selected_muon_etas, d_selected_muon_phis,
  // d_new_muon_offsets, d_muon_segment_mask);

  // The transform version seems to perform better. Uncommenting out the segmented reduce version yields the same result
  // with different performance.
  auto masses_electrons =
    invariant_mass_transform(d_final_electron_pts, d_final_electron_etas, d_final_electron_phis, segment_size);
  auto masses_muons = invariant_mass_transform(d_final_muon_pts, d_final_muon_etas, d_final_muon_phis, segment_size);
  // auto masses_electrons = invariant_mass_segmented_reduce(d_electron_pts, d_electron_etas, d_electron_phis,
  // segment_size); auto masses_muons     = invariant_mass_segmented_reduce(d_muon_pts, d_muon_etas, d_muon_phis,
  // segment_size);

  return {masses_electrons, masses_muons};
}

template <typename T>
static void physics_analysis(nvbench::state& state, nvbench::type_list<T>)
{
  const auto num_events = static_cast<std::size_t>(state.get_int64("Events{io}"));

  // Use typed literals to let compiler deduce template argument (avoids nvbench_helper bug)
  thrust::device_vector<int> num_electrons_per_event = generate(num_events, bit_entropy::_1_000, int{0}, int{10});
  thrust::device_vector<int> num_muons_per_event     = generate(num_events, bit_entropy::_1_000, int{0}, int{10});

  thrust::device_vector<int> electron_offsets(num_events + 1, thrust::no_init);
  thrust::device_vector<int> muon_offsets(num_events + 1, thrust::no_init);

  thrust::exclusive_scan(num_electrons_per_event.begin(), num_electrons_per_event.end(), electron_offsets.begin(), 0);
  electron_offsets[num_events] = electron_offsets[num_events - 1] + num_electrons_per_event[num_events - 1];

  thrust::exclusive_scan(num_muons_per_event.begin(), num_muons_per_event.end(), muon_offsets.begin(), 0);
  muon_offsets[num_events] = muon_offsets[num_events - 1] + num_muons_per_event[num_events - 1];

  int total_electrons = electron_offsets[num_events];
  int total_muons     = muon_offsets[num_events];

  thrust::device_vector<T> electron_pts  = generate(total_electrons, bit_entropy::_1_000, T{10.0}, T{100.0});
  thrust::device_vector<T> electron_etas = generate(total_electrons, bit_entropy::_1_000, T{-3.0}, T{3.0});
  thrust::device_vector<T> electron_phis = generate(total_electrons, bit_entropy::_1_000, T{0.0}, T{2.0 * M_PI});

  thrust::device_vector<T> muons_pts  = generate(total_muons, bit_entropy::_1_000, T{10.0}, T{100.0});
  thrust::device_vector<T> muons_etas = generate(total_muons, bit_entropy::_1_000, T{-3.0}, T{3.0});
  thrust::device_vector<T> muons_phis = generate(total_muons, bit_entropy::_1_000, T{0.0}, T{2.0 * M_PI});

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    physics_analysis(
      electron_pts, electron_etas, electron_phis, muons_pts, muons_etas, muons_phis, electron_offsets, muon_offsets);
  });
}

using current_data_types = nvbench::type_list<float, double>;

NVBENCH_BENCH_TYPES(physics_analysis, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("physics_analysis")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Events{io}", nvbench::range(12, 24, 4));
