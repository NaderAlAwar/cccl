#include <nvbench_helper.cuh>

#include "bench_util.cuh"
#include "invariant_mass_segmented_reduce.cuh"
#include "invariant_mass_transform.cuh"

// Step 1: Implement using device transform
template <typename T>
static void invariant_mass_transform(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
  // Example: [[30,40] [20,50], [10,30]]
  //          [[2.1, 2.2], [-1.5,3], [0.4,3.7]]
  //          [[0.3,0.5], [0.6,0.3], [0.1,0.7]]
  // Flatten:
  thrust::device_vector<T> d_pt{30, 40, 20, 50, 10, 30};
  thrust::device_vector<T> d_eta{2.1, 2.2, -1.5, 3, 0.4, 3.7};
  thrust::device_vector<T> d_phi{0.3, 0.5, 0.6, 0.3, 0.1, 0.7};
  thrust::device_vector<int> d_offsets{0, 2, 4, 6}; // 3 segments
  constexpr int segment_size = 2;
  std::cout << "Before calculation:" << std::endl;
  print_array(d_pt, d_offsets);
  print_array(d_eta, d_offsets);
  print_array(d_phi, d_offsets);

  auto d_result = invariant_mass_transform(d_pt, d_eta, d_phi, segment_size);

  std::cout << "Result:" << std::endl;
  print_array(d_result);
#else
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr int segment_size = 2;

  thrust::device_vector<T> d_pt  = generate(elements);
  thrust::device_vector<T> d_eta = generate(elements);
  thrust::device_vector<T> d_phi = generate(elements);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    invariant_mass_transform(d_pt, d_eta, d_phi, segment_size);
  });
#endif
}

// Step 2: Implement using device segmented reduce
template <typename T>
static void invariant_mass_segmented_reduce(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
  // Example: [[30,40] [20,50], [10,30]]
  //          [[2.1, 2.2], [-1.5,3], [0.4,3.7]]
  //          [[0.3,0.5], [0.6,0.3], [0.1,0.7]]
  // Flatten:
  thrust::device_vector<T> d_pt{30, 40, 20, 50, 10, 30};
  thrust::device_vector<T> d_eta{2.1, 2.2, -1.5, 3, 0.4, 3.7};
  thrust::device_vector<T> d_phi{0.3, 0.5, 0.6, 0.3, 0.1, 0.7};
  thrust::device_vector<int> d_offsets{0, 2, 4, 6}; // 3 segments
  constexpr int segment_size = 2;
  std::cout << "Before calculation:" << std::endl;
  print_array(d_pt, d_offsets);
  print_array(d_eta, d_offsets);
  print_array(d_phi, d_offsets);

  auto d_result = invariant_mass_segmented_reduce(d_pt, d_eta, d_phi, segment_size);

  std::cout << "Result:" << std::endl;
  print_array(d_result);
#else
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  constexpr int segment_size = 2;

  thrust::device_vector<T> d_pt  = generate(elements);
  thrust::device_vector<T> d_eta = generate(elements);
  thrust::device_vector<T> d_phi = generate(elements);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    invariant_mass_segmented_reduce(d_pt, d_eta, d_phi, segment_size);
  });
#endif
}

using current_data_types = nvbench::type_list<float>;

NVBENCH_BENCH_TYPES(invariant_mass_transform, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("invariant_mass_transform")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 28, 4));

NVBENCH_BENCH_TYPES(invariant_mass_segmented_reduce, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("invariant_mass_segmented_reduce")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 28, 4));
