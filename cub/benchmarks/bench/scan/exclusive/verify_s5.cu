// This file is used to verify the S5 scan implementation. It reads the input
// data from .npy files and runs both CUB scan operations. It then writes the
// outputs to .npy files.

#include <vector>

#include "libnpy/include/npy.hpp"
#include "s5_operator.cuh"
#include "s5_operator_segmented.cuh"
#include "s5_operator_segmented_actual.cuh"
#include <nvbench/nvbench.cuh>

using DataType          = float;
constexpr int state_dim = 40;
// constexpr int state_dim = 3;

void read_npy_file(const std::string& filename, std::vector<DataType>& data)
{
  npy::npy_data<DataType> npy_data = npy::read_npy<DataType>(filename);
  data                             = npy_data.data;
}

void write_npy_file(
  const std::string& filename, const std::vector<DataType>& data, const std::vector<unsigned long>& shape)
{
  npy::npy_data<DataType> npy_data;
  npy_data.data          = data;
  npy_data.shape         = shape;
  npy_data.fortran_order = false;
  npy::write_npy(filename, npy_data);
}

void s5_scan_verify(nvbench::state& state)
{
  // Read input data
  std::vector<DataType> h_A_in, h_Bu_in;

  read_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/A_in.npy", h_A_in);
  read_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/Bu_in.npy", h_Bu_in);

  const int timesteps = h_A_in.size() / state_dim;

  thrust::device_vector<DataType> d_A_in(h_A_in);
  thrust::device_vector<DataType> d_Bu_in(h_Bu_in);

  size_t temp_storage_bytes = 0;
  auto input_iter = s5_operator::setup_scan<DataType, state_dim>(d_A_in, d_Bu_in, timesteps, temp_storage_bytes);

  thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  thrust::device_vector<DataType> d_A_out(timesteps * state_dim, thrust::no_init);
  thrust::device_vector<DataType> d_Bu_out(timesteps * state_dim, thrust::no_init);

  // Run once to verify
  state.exec([&](nvbench::launch& launch) {
    cudaStream_t stream = launch.get_stream();
    s5_operator::run_scan<DataType, state_dim>(
      d_temp_storage, temp_storage_bytes, input_iter, d_A_out, d_Bu_out, timesteps, stream);
  });

  thrust::host_vector<DataType> h_A_out  = d_A_out;
  thrust::host_vector<DataType> h_Bu_out = d_Bu_out;

  // Write output
  write_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/A_out_cpp_s5.npy",
                 std::vector<DataType>(h_A_out.begin(), h_A_out.end()),
                 {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(state_dim)});
  write_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/Bu_out_cpp_s5.npy",
                 std::vector<DataType>(h_Bu_out.begin(), h_Bu_out.end()),
                 {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(state_dim)});
}

void s5_segmented_scan_verify(nvbench::state& state)
{
  // Read input data
  std::vector<DataType> h_A_in, h_Bu_in;

  read_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/A_in.npy", h_A_in);
  read_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/Bu_in.npy", h_Bu_in);

  const int timesteps = h_A_in.size() / state_dim;

  thrust::device_vector<DataType> d_A_in(h_A_in);
  thrust::device_vector<DataType> d_Bu_in(h_Bu_in);

  size_t temp_storage_bytes = 0;
  auto input_iter =
    s5_operator_segmented::setup_scan<DataType, state_dim>(d_A_in, d_Bu_in, timesteps, temp_storage_bytes);

  thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  thrust::device_vector<DataType> d_A_out(timesteps * state_dim, thrust::no_init);
  thrust::device_vector<DataType> d_Bu_out(timesteps * state_dim, thrust::no_init);

  // Run once to verify
  state.exec([&](nvbench::launch& launch) {
    cudaStream_t stream = launch.get_stream();
    s5_operator_segmented::run_scan<DataType, state_dim>(
      d_temp_storage, temp_storage_bytes, input_iter, d_A_out, d_Bu_out, timesteps, stream);
  });

  thrust::host_vector<DataType> h_A_out  = d_A_out;
  thrust::host_vector<DataType> h_Bu_out = d_Bu_out;

  // Write output
  write_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/A_out_cpp_s5_segmented.npy",
                 std::vector<DataType>(h_A_out.begin(), h_A_out.end()),
                 {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(state_dim)});
  write_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/Bu_out_cpp_s5_segmented.npy",
                 std::vector<DataType>(h_Bu_out.begin(), h_Bu_out.end()),
                 {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(state_dim)});
}

void s5_segmented_scan_actual_verify(nvbench::state& state)
{
  // Read input data
  std::vector<DataType> h_A_in, h_Bu_in;

  read_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/A_in.npy", h_A_in);
  read_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/Bu_in.npy", h_Bu_in);

  const int timesteps = h_A_in.size() / state_dim;

  thrust::device_vector<DataType> d_A_in(h_A_in);
  thrust::device_vector<DataType> d_Bu_in(h_Bu_in);

  size_t temp_storage_bytes = 0;
  auto [input_iter, begin_offsets, end_offsets] =
    s5_operator_segmented_actual::setup_scan<DataType, state_dim>(d_A_in, d_Bu_in, timesteps, temp_storage_bytes);

  thrust::device_vector<nvbench::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
  void* d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  thrust::device_vector<DataType> d_A_out(timesteps * state_dim, thrust::no_init);
  thrust::device_vector<DataType> d_Bu_out(timesteps * state_dim, thrust::no_init);

  // Run once to verify
  state.exec([&](nvbench::launch& launch) {
    cudaStream_t stream = launch.get_stream();
    s5_operator_segmented_actual::run_scan<DataType, state_dim>(
      d_temp_storage, temp_storage_bytes, input_iter, d_A_out, d_Bu_out, begin_offsets, end_offsets, timesteps, stream);
  });

  thrust::host_vector<DataType> h_A_out  = d_A_out;
  thrust::host_vector<DataType> h_Bu_out = d_Bu_out;

  // Write output
  write_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/A_out_cpp_s5_segmented_actual.npy",
                 std::vector<DataType>(h_A_out.begin(), h_A_out.end()),
                 {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(state_dim)});
  write_npy_file("/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20/Bu_out_cpp_s5_segmented_actual.npy",
                 std::vector<DataType>(h_Bu_out.begin(), h_Bu_out.end()),
                 {static_cast<unsigned long>(timesteps), static_cast<unsigned long>(state_dim)});
}

NVBENCH_BENCH(s5_scan_verify).set_name("s5_scan_verify");

NVBENCH_BENCH(s5_segmented_scan_verify).set_name("s5_segmented_scan_verify");

NVBENCH_BENCH(s5_segmented_scan_actual_verify).set_name("s5_segmented_scan_verify");
