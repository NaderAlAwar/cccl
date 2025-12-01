// This file implements the end to end benchmark needed by awkward. This includes three steps:
// 1) A segmented filter implementation
// 2) Filtering out segments based on length
// 3) Invariant mass calculation

#include <thrust/copy.h>

#include <cuda/std/numeric>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvbench_helper.cuh>

#include "filter_out_segments_copy_if_zipped.cuh"
#include "filter_out_segments_fancy_iterator_zipped.cuh"
#include "filter_out_segments_rle_scan_zipped.cuh"
#include "filter_segmented_array_upper_bound_zipped.cuh"
#include "filter_segmented_array_zipped.cuh"
#include "invariant_mass_segmented_reduce.cuh"
#include "invariant_mass_transform.cuh"
#include "libnpy/include/npy.hpp"

namespace
{
namespace fs = std::filesystem;

template <typename T>
std::vector<T> load_npy_vector(const fs::path& file_path)
{
  auto array = npy::read_npy<T>(file_path.string());
  if (array.fortran_order && array.shape.size() > 1)
  {
    throw std::runtime_error("Fortran-ordered arrays are not supported for " + file_path.string());
  }
  return std::move(array.data);
}

template <typename T>
std::vector<T> copy_device_to_host(const thrust::device_vector<T>& device_vec)
{
  std::vector<T> host(device_vec.size());
  thrust::copy(device_vec.begin(), device_vec.end(), host.begin());
  return host;
}

template <typename T>
void write_npy_vector(const std::vector<T>& host, const fs::path& file_path)
{
  npy::npy_data<T> npy_buffer;
  npy_buffer.shape         = {static_cast<npy::ndarray_len_t>(host.size())};
  npy_buffer.fortran_order = false;
  npy_buffer.data          = host;
  npy::write_npy(file_path.string(), npy_buffer);
}
} // namespace

template <typename T>
static void physics_analysis(
  const thrust::device_vector<T>& d_electron_pts,
  const thrust::device_vector<T>& d_electron_etas,
  const thrust::device_vector<T>& d_electron_phis,
  const thrust::device_vector<T>& d_muon_pts,
  const thrust::device_vector<T>& d_muon_etas,
  const thrust::device_vector<T>& d_muon_phis,
  const thrust::device_vector<int>& d_electron_offsets,
  const thrust::device_vector<int>& d_muon_offsets,
  thrust::device_vector<T>& d_temp_electron_pts,
  thrust::device_vector<T>& d_temp_electron_etas,
  thrust::device_vector<T>& d_temp_electron_phis,
  thrust::device_vector<T>& d_temp_muon_pts,
  thrust::device_vector<T>& d_temp_muon_etas,
  thrust::device_vector<T>& d_temp_muon_phis,
  thrust::device_vector<int>& d_temp_electron_offsets,
  thrust::device_vector<int>& d_temp_muon_offsets,
  thrust::device_vector<bool>& d_electron_segment_mask,
  thrust::device_vector<bool>& d_muon_segment_mask,
  thrust::device_vector<int>& d_temp_electron_num_removed_per_segment,
  thrust::device_vector<int>& d_temp_muon_num_removed_per_segment,
  thrust::device_vector<T>& d_temp2_electron_pts,
  thrust::device_vector<T>& d_temp2_electron_etas,
  thrust::device_vector<T>& d_temp2_electron_phis,
  thrust::device_vector<T>& d_temp2_muon_pts,
  thrust::device_vector<T>& d_temp2_muon_etas,
  thrust::device_vector<T>& d_temp2_muon_phis,
  thrust::device_vector<int>& d_temp2_electron_segment_ids,
  thrust::device_vector<int>& d_temp2_muon_segment_ids,
  thrust::device_vector<int>& d_temp2_electron_num_selected_out,
  thrust::device_vector<int>& d_temp2_muon_num_selected_out,
  thrust::device_vector<T>& d_masses_electrons,
  thrust::device_vector<T>& d_masses_muons,
  thrust::device_vector<uint8_t>& d_temp_storage)
{
  auto cond_electron = [] __device__(const cuda::std::tuple<T, T, T>& x) {
    return cuda::std::get<0>(x) > 40.0;
  };
  auto cond_muon = [] __device__(const cuda::std::tuple<T, T, T>& x) {
    return cuda::std::get<0>(x) > 20.0 && cuda::std::abs(cuda::std::get<1>(x)) < 2.4;
  };

  // The stateful op version seems to perform better. Uncommenting out the upper bound version yields the same result
  // with different performance.
  segmented_filter_zipped(
    d_electron_pts,
    d_electron_etas,
    d_electron_phis,
    d_electron_offsets,
    d_temp_electron_pts,
    d_temp_electron_etas,
    d_temp_electron_phis,
    d_temp_electron_offsets,
    d_temp2_electron_num_selected_out,
    d_temp_electron_num_removed_per_segment,
    d_temp_storage,
    cond_electron);
  segmented_filter_zipped(
    d_muon_pts,
    d_muon_etas,
    d_muon_phis,
    d_muon_offsets,
    d_temp_muon_pts,
    d_temp_muon_etas,
    d_temp_muon_phis,
    d_temp_muon_offsets,
    d_temp2_muon_num_selected_out,
    d_temp_muon_num_removed_per_segment,
    d_temp_storage,
    cond_muon);
  // segmented_filter_upper_bound_zipped(
  //   d_electron_pts,
  //   d_electron_etas,
  //   d_electron_phis,
  //   d_electron_offsets,
  //   d_temp_electron_pts,
  //   d_temp_electron_etas,
  //   d_temp_electron_phis,
  //   d_temp_electron_offsets,
  //   d_temp2_electron_num_selected_out,
  //   d_temp_electron_num_removed_per_segment,
  //   d_temp_storage,
  //   cond_electron);
  // segmented_filter_upper_bound_zipped(
  //   d_muon_pts,
  //   d_muon_etas,
  //   d_muon_phis,
  //   d_muon_offsets,
  //   d_temp_muon_pts,
  //   d_temp_muon_etas,
  //   d_temp_muon_phis,
  //   d_temp_muon_offsets,
  //   d_temp2_muon_num_selected_out,
  //   d_temp_muon_num_removed_per_segment,
  //   d_temp_storage,
  //   cond_muon);

  const auto num_electron_segments = d_temp_electron_offsets.size() - 1;
  const auto num_muon_segments     = d_temp_muon_offsets.size() - 1;

  constexpr int segment_size = 2;

  auto error = cub::DeviceTransform::Transform(
    cuda::std::tuple{d_temp_electron_offsets.begin(), d_temp_electron_offsets.begin() + 1},
    thrust::raw_pointer_cast(d_electron_segment_mask.data()),
    num_electron_segments,
    [segment_size] __device__(int start, int end) {
      return (end - start) == segment_size;
    });

  if (error != cudaSuccess)
  {
    std::cerr << "Error during electron segment length calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  error = cub::DeviceTransform::Transform(
    cuda::std::tuple{d_temp_muon_offsets.begin(), d_temp_muon_offsets.begin() + 1},
    thrust::raw_pointer_cast(d_muon_segment_mask.data()),
    num_muon_segments,
    [segment_size] __device__(int start, int end) {
      return (end - start) == segment_size;
    });

  if (error != cudaSuccess)
  {
    std::cerr << "Error during muon segment length calculation: " << cudaGetErrorString(error) << std::endl;
    return;
  }

  // The fancy iterator version seems to perform better. Uncommenting out the copy_if and rle_scan versions yields the
  // same result with different performance.
  filter_out_segments_fancy_iterator_zipped(
    d_temp_electron_pts,
    d_temp_electron_etas,
    d_temp_electron_phis,
    d_temp_electron_offsets,
    d_temp2_electron_pts,
    d_temp2_electron_etas,
    d_temp2_electron_phis,
    d_temp2_electron_segment_ids,
    d_temp2_electron_num_selected_out,
    d_temp_storage,
    d_electron_segment_mask);
  filter_out_segments_fancy_iterator_zipped(
    d_temp_muon_pts,
    d_temp_muon_etas,
    d_temp_muon_phis,
    d_temp_muon_offsets,
    d_temp2_muon_pts,
    d_temp2_muon_etas,
    d_temp2_muon_phis,
    d_temp2_muon_segment_ids,
    d_temp2_muon_num_selected_out,
    d_temp_storage,
    d_muon_segment_mask);
  // filter_out_segments_copy_if_zipped(d_temp_electron_pts, d_temp_electron_etas, d_temp_electron_phis,
  // d_temp_electron_offsets, d_electron_segment_mask); filter_out_segments_copy_if_zipped(d_temp_muon_pts,
  // d_temp_muon_etas, d_temp_muon_phis, d_temp_muon_offsets, d_muon_segment_mask);
  // filter_out_segments_rle_scan_zipped(d_temp_electron_pts, d_temp_electron_etas, d_temp_electron_phis,
  // d_temp_electron_offsets, d_electron_segment_mask); filter_out_segments_rle_scan_zipped(d_temp_muon_pts,
  // d_temp_muon_etas, d_temp_muon_phis, d_temp_muon_offsets, d_muon_segment_mask);

  // // The transform version seems to perform better. Uncommenting out the segmented reduce version yields the same
  // result
  // // with different performance.
  invariant_mass_transform(
    d_temp2_electron_pts, d_temp2_electron_etas, d_temp2_electron_phis, d_masses_electrons, segment_size);
  invariant_mass_transform(d_temp2_muon_pts, d_temp2_muon_etas, d_temp2_muon_phis, d_masses_muons, segment_size);
  // // auto masses_electrons = invariant_mass_segmented_reduce(d_temp_electron_pts, d_temp_electron_etas,
  // // d_temp_electron_phis, segment_size); auto masses_muons     = invariant_mass_segmented_reduce(d_temp_muon_pts,
  // // d_temp_muon_etas, d_temp_muon_phis, segment_size);
}

template <typename T>
static void physics_analysis(nvbench::state& state, nvbench::type_list<T>)
{
  bool check_correctness = true;
  thrust::device_vector<T> d_electron_pts;
  thrust::device_vector<T> d_electron_etas;
  thrust::device_vector<T> d_electron_phis;
  thrust::device_vector<T> d_muons_pts;
  thrust::device_vector<T> d_muons_etas;
  thrust::device_vector<T> d_muons_phis;
  thrust::device_vector<int> d_electron_offsets;
  thrust::device_vector<int> d_muon_offsets;

  fs::path repo_root{"/home/coder/cccl"};

  if (check_correctness)
  {
    try
    {
      const auto power_of_10   = static_cast<int>(state.get_int64("10Power{io}"));
      const std::string suffix = "_" + std::to_string(power_of_10);

      const auto build_path = [&](const std::string& name) {
        return repo_root / (name + suffix + ".npy");
      };

      auto electron_pts_host     = load_npy_vector<T>(build_path("electron_pts"));
      auto electron_etas_host    = load_npy_vector<T>(build_path("electron_etas"));
      auto electron_phis_host    = load_npy_vector<T>(build_path("electron_phis"));
      auto muons_pts_host        = load_npy_vector<T>(build_path("muons_pts"));
      auto muons_etas_host       = load_npy_vector<T>(build_path("muons_etas"));
      auto muons_phis_host       = load_npy_vector<T>(build_path("muons_phis"));
      auto electron_offsets_host = load_npy_vector<int>(build_path("electron_offsets"));
      auto muon_offsets_host     = load_npy_vector<int>(build_path("muon_offsets"));

      const auto require = [](bool condition, const std::string& message) {
        if (!condition)
        {
          throw std::runtime_error(message);
        }
      };

      require(electron_pts_host.size() == electron_etas_host.size(), "electron eta array length mismatch");
      require(electron_pts_host.size() == electron_phis_host.size(), "electron phi array length mismatch");
      require(muons_pts_host.size() == muons_etas_host.size(), "muon eta array length mismatch");
      require(muons_pts_host.size() == muons_phis_host.size(), "muon phi array length mismatch");
      require(!electron_offsets_host.empty(), "electron offsets array cannot be empty");
      require(!muon_offsets_host.empty(), "muon offsets array cannot be empty");

      const auto expected_electron_count = static_cast<std::size_t>(electron_offsets_host.back());
      const auto expected_muon_count     = static_cast<std::size_t>(muon_offsets_host.back());

      require(expected_electron_count == electron_pts_host.size(), "electron offsets total does not match data length");
      require(expected_muon_count == muons_pts_host.size(), "muon offsets total does not match data length");

      d_electron_pts.assign(electron_pts_host.begin(), electron_pts_host.end());
      d_electron_etas.assign(electron_etas_host.begin(), electron_etas_host.end());
      d_electron_phis.assign(electron_phis_host.begin(), electron_phis_host.end());
      d_muons_pts.assign(muons_pts_host.begin(), muons_pts_host.end());
      d_muons_etas.assign(muons_etas_host.begin(), muons_etas_host.end());
      d_muons_phis.assign(muons_phis_host.begin(), muons_phis_host.end());
      d_electron_offsets.assign(electron_offsets_host.begin(), electron_offsets_host.end());
      d_muon_offsets.assign(muon_offsets_host.begin(), muon_offsets_host.end());

      // Warm up CUDA context to ensure device code is loaded
      cudaFree(nullptr);
    }
    catch (const std::exception& ex)
    {
      std::cerr << "Failed to load Awkward benchmark data from disk: " << ex.what() << std::endl;
      return;
    }
  }
  else
  {
    const auto num_events = static_cast<std::size_t>(state.get_int64("Events{io}"));

    // Use typed literals to let compiler deduce template argument (avoids nvbench_helper bug)
    thrust::device_vector<int> num_electrons_per_event = generate(num_events, bit_entropy::_1_000, int{0}, int{10});
    thrust::device_vector<int> num_muons_per_event     = generate(num_events, bit_entropy::_1_000, int{0}, int{10});

    d_electron_offsets.resize(num_events + 1);
    d_muon_offsets.resize(num_events + 1);

    thrust::exclusive_scan(
      num_electrons_per_event.begin(), num_electrons_per_event.end(), d_electron_offsets.begin(), 0);
    d_electron_offsets[num_events] = d_electron_offsets[num_events - 1] + num_electrons_per_event[num_events - 1];

    thrust::exclusive_scan(num_muons_per_event.begin(), num_muons_per_event.end(), d_muon_offsets.begin(), 0);
    d_muon_offsets[num_events] = d_muon_offsets[num_events - 1] + num_muons_per_event[num_events - 1];

    const int total_electrons = d_electron_offsets[num_events];
    const int total_muons     = d_muon_offsets[num_events];

    d_electron_pts  = generate(total_electrons, bit_entropy::_1_000, T{10.0}, T{100.0});
    d_electron_etas = generate(total_electrons, bit_entropy::_1_000, T{-3.0}, T{3.0});
    d_electron_phis = generate(total_electrons, bit_entropy::_1_000, T{0.0}, T{2.0 * M_PI});

    d_muons_pts  = generate(total_muons, bit_entropy::_1_000, T{10.0}, T{100.0});
    d_muons_etas = generate(total_muons, bit_entropy::_1_000, T{-3.0}, T{3.0});
    d_muons_phis = generate(total_muons, bit_entropy::_1_000, T{0.0}, T{2.0 * M_PI});
  }

  // In order to do a fair comparison with cupy, which allocates these
  // arrays inside the benchmark using a caching allocator, we need to
  // pre-allocate all the temporary intermediate arrays here.
  thrust::device_vector<T> d_temp_electron_pts(d_electron_pts.size(), thrust::no_init);
  thrust::device_vector<T> d_temp_electron_etas(d_electron_etas.size(), thrust::no_init);
  thrust::device_vector<T> d_temp_electron_phis(d_electron_phis.size(), thrust::no_init);
  thrust::device_vector<T> d_temp_muon_pts(d_muons_pts.size(), thrust::no_init);
  thrust::device_vector<T> d_temp_muon_etas(d_muons_etas.size(), thrust::no_init);
  thrust::device_vector<T> d_temp_muon_phis(d_muons_phis.size(), thrust::no_init);
  thrust::device_vector<int> d_temp_electron_offsets(d_electron_offsets.size(), thrust::no_init);
  thrust::device_vector<int> d_temp_muon_offsets(d_muon_offsets.size(), thrust::no_init);
  thrust::device_vector<bool> d_electron_segment_mask(d_electron_offsets.size() - 1, thrust::no_init);
  thrust::device_vector<bool> d_muon_segment_mask(d_muon_offsets.size() - 1, thrust::no_init);
  thrust::device_vector<int> d_temp_electron_num_removed_per_segment(d_electron_offsets.size() - 1, 0);
  thrust::device_vector<int> d_temp_muon_num_removed_per_segment(d_muon_offsets.size() - 1, 0);
  thrust::device_vector<T> d_masses_electrons(d_electron_pts.size(), thrust::no_init);
  thrust::device_vector<T> d_masses_muons(d_muons_pts.size(), thrust::no_init);
  thrust::device_vector<T> d_temp2_electron_pts(d_electron_pts.size(), thrust::no_init);
  thrust::device_vector<T> d_temp2_electron_etas(d_electron_etas.size(), thrust::no_init);
  thrust::device_vector<T> d_temp2_electron_phis(d_electron_phis.size(), thrust::no_init);
  thrust::device_vector<T> d_temp2_muon_pts(d_muons_pts.size(), thrust::no_init);
  thrust::device_vector<T> d_temp2_muon_etas(d_muons_etas.size(), thrust::no_init);
  thrust::device_vector<T> d_temp2_muon_phis(d_muons_phis.size(), thrust::no_init);
  thrust::device_vector<int> d_temp2_electron_segment_ids(d_electron_pts.size(), thrust::no_init);
  thrust::device_vector<int> d_temp2_muon_segment_ids(d_muons_pts.size(), thrust::no_init);
  thrust::device_vector<int> d_temp2_electron_num_selected_out(1, thrust::no_init);
  thrust::device_vector<int> d_temp2_muon_num_selected_out(1, thrust::no_init);
  // Rounded this value up from 4135679. Found it by printing the
  // temp_storage_bytes value from a previous run.
  thrust::device_vector<uint8_t> d_temp_storage(5000000, thrust::no_init);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    // Reset array sizes before each iteration since the functions resize them
    d_temp_electron_pts.resize(d_electron_pts.size());
    d_temp_electron_etas.resize(d_electron_etas.size());
    d_temp_electron_phis.resize(d_electron_phis.size());
    d_temp_muon_pts.resize(d_muons_pts.size());
    d_temp_muon_etas.resize(d_muons_etas.size());
    d_temp_muon_phis.resize(d_muons_phis.size());
    d_temp_electron_offsets.resize(d_electron_offsets.size());
    d_temp_muon_offsets.resize(d_muon_offsets.size());
    d_temp2_electron_pts.resize(d_electron_pts.size());
    d_temp2_electron_etas.resize(d_electron_etas.size());
    d_temp2_electron_phis.resize(d_electron_phis.size());
    d_temp2_muon_pts.resize(d_muons_pts.size());
    d_temp2_muon_etas.resize(d_muons_etas.size());
    d_temp2_muon_phis.resize(d_muons_phis.size());
    d_masses_electrons.resize(d_electron_pts.size());
    d_masses_muons.resize(d_muons_pts.size());
    d_electron_segment_mask.resize(d_electron_offsets.size() - 1);
    d_muon_segment_mask.resize(d_muon_offsets.size() - 1);
    d_temp_electron_num_removed_per_segment.assign(d_electron_offsets.size() - 1, 0);
    d_temp_muon_num_removed_per_segment.assign(d_muon_offsets.size() - 1, 0);

    physics_analysis(
      d_electron_pts,
      d_electron_etas,
      d_electron_phis,
      d_muons_pts,
      d_muons_etas,
      d_muons_phis,
      d_electron_offsets,
      d_muon_offsets,
      d_temp_electron_pts,
      d_temp_electron_etas,
      d_temp_electron_phis,
      d_temp_muon_pts,
      d_temp_muon_etas,
      d_temp_muon_phis,
      d_temp_electron_offsets,
      d_temp_muon_offsets,
      d_electron_segment_mask,
      d_muon_segment_mask,
      d_temp_electron_num_removed_per_segment,
      d_temp_muon_num_removed_per_segment,
      d_temp2_electron_pts,
      d_temp2_electron_etas,
      d_temp2_electron_phis,
      d_temp2_muon_pts,
      d_temp2_muon_etas,
      d_temp2_muon_phis,
      d_temp2_electron_segment_ids,
      d_temp2_muon_segment_ids,
      d_temp2_electron_num_selected_out,
      d_temp2_muon_num_selected_out,
      d_masses_electrons,
      d_masses_muons,
      d_temp_storage);
  });

  if (check_correctness)
  {
    const auto power_of_10   = static_cast<int>(state.get_int64("10Power{io}"));
    const std::string suffix = "_" + std::to_string(power_of_10);

    const auto build_output_path = [&](const std::string& name) {
      return repo_root / (name + suffix + "_cpp.npy");
    };
    const auto masses_electrons_host = copy_device_to_host(d_masses_electrons);
    const auto masses_muons_host     = copy_device_to_host(d_masses_muons);

    write_npy_vector(masses_electrons_host, build_output_path("masses_electrons"));
    write_npy_vector(masses_muons_host, build_output_path("masses_muons"));
  }
}

// using current_data_types = nvbench::type_list<float, double>;

// NVBENCH_BENCH_TYPES(physics_analysis, NVBENCH_TYPE_AXES(current_data_types))
//   .set_name("physics_analysis")
//   .set_type_axes_names({"T{ct}"})
//   .add_int64_power_of_two_axis("Events{io}", nvbench::range(12, 24, 4));

using current_data_types = nvbench::type_list<double>;

NVBENCH_BENCH_TYPES(physics_analysis, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("physics_analysis")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("10Power{io}", nvbench::range(4, 7, 1));
