// This attempts to create a CUB implementation of the functionality used by
// awkward here
// https:github.com/scikit-hep/awkward/blob/af08e7b5bd07e81e81a2c4b5a72135a3c5c5b18b/studies/cccl/playground.py#L38

// The benchmark should demonstrate this functionality:

// Given an input zip iterator of three segmented arrays, each with a data
// buffer and an offsets buffer, filter the elements of the data buffers
// according to some condition (e.g., value > threshold), producing output data
// buffers and offsets buffers. The tricky part is that this will modify those
// arrays so the output offsets will differ from the input offsets.

// For now I will think about how to do this for a single segmented array
// instead of the zipped three arrays.

#include <random>

#include <nvbench_helper.cuh>

#include "filter_flat_array.cuh"
#include "filter_segmented_array.cuh"

template <typename T>
static void print_array(const thrust::device_vector<T>& d_values)
{
  thrust::host_vector<int> h_values = d_values;

  for (T x : h_values)
  {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

template <typename T>
static void print_array(const thrust::device_vector<T>& d_values, const thrust::device_vector<int>& d_offsets)
{
  thrust::host_vector<T> h_values    = d_values;
  thrust::host_vector<int> h_offsets = d_offsets;

  int num_segments = static_cast<int>(h_offsets.size()) - 1;

  for (int seg = 0; seg < num_segments; ++seg)
  {
    int start = h_offsets[seg];
    int end   = h_offsets[seg + 1];

    std::cout << "Segment " << seg << ": ";
    for (int i = start; i < end; ++i)
    {
      std::cout << h_values[i] << " ";
    }
    std::cout << std::endl;
  }
}

// Step 1: Filter out a single array
template <typename T>
static void filter(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
  // Example: [[30, 40, 20, 50, 10, 30, 80]]
  thrust::device_vector<T> d_values{30, 40, 20, 50, 10, 30, 80};
  constexpr T threshold = 25;

  std::cout << "Running single array filter sample:" << std::endl;
  std::cout << "Before filtering:" << std::endl;
  print_array(d_values);
  filter(d_values, threshold);
  std::cout << "After filtering (threshold = " << threshold << "):" << std::endl;
  print_array(d_values);
#else
  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<T> d_values = generate(elements);
  const T threshold                 = lerp_min_max<T>(entropy_to_probability(entropy));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    filter(d_values, threshold);
  });

#endif
}

thrust::host_vector<int> partition_into_segments(size_t num_elements, int max_segment_size)
{
  thrust::host_vector<int> sizes;

  std::random_device rd;
  std::mt19937 gen(rd());

  int remaining = num_elements;
  while (remaining > 0)
  {
    int max_size = std::min(max_segment_size, remaining);
    std::uniform_int_distribution<int> dist(1, max_size);
    int size = dist(gen);

    sizes.push_back(size);
    remaining -= size;
  }

  return sizes;
}

// Step 2: Extend to a segmented array
template <typename T>
static void segmented_filter(nvbench::state& state, nvbench::type_list<T>)
{
#if RUN_SAMPLE
  // Example: [[30], [40,20], [50], [10,30,80]]
  // Flatten:
  thrust::device_vector<T> d_values{30, 40, 20, 50, 10, 30, 80};
  thrust::device_vector<int> d_offsets{0, 1, 3, 4, 7}; // 4 segments
  constexpr T threshold = 25;

  std::cout << "Running segmented array filter sample:" << std::endl;
  std::cout << "Before filtering:" << std::endl;
  print_array(d_values, d_offsets);
  segmented_filter(d_values, d_offsets, threshold);
  std::cout << "After filtering (threshold = " << threshold << "):" << std::endl;
  print_array(d_values, d_offsets);
#else
  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  constexpr int max_segment_size = 20;
  const int max_num_segments     = cuda::ceil_div(elements, max_segment_size);

  thrust::device_vector<T> d_values = generate(elements);

  auto segment_sizes = partition_into_segments(elements, max_segment_size);
  thrust::host_vector<int> h_offsets(segment_sizes.size() + 1, 0);
  std::exclusive_scan(segment_sizes.begin(), segment_sizes.end(), h_offsets.begin() + 1, 0);
  h_offsets[segment_sizes.size()] = segment_sizes[segment_sizes.size() - 1] + segment_sizes[segment_sizes.size() - 1];

  thrust::device_vector<int> d_offsets = h_offsets;
  const T threshold                    = lerp_min_max<T>(entropy_to_probability(entropy));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    segmented_filter(d_values, d_offsets, threshold);
  });

#endif
}

using current_data_types = nvbench::type_list<int>;

NVBENCH_BENCH_TYPES(filter, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 24, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});

NVBENCH_BENCH_TYPES(segmented_filter, NVBENCH_TYPE_AXES(current_data_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(12, 24, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
