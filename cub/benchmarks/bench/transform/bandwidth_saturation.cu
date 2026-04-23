// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>
#include <nvbench_helper.cuh>

namespace
{
using value_t = nvbench::uint32_t;

constexpr std::size_t kib = 1024;
constexpr std::size_t mib = 1024 * kib;

constexpr std::size_t min_input_bytes = 128 * mib;
constexpr std::size_t max_input_bytes = 1024 * mib;

template <int LoadsPerThread, int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void streaming_read_kernel(
  const value_t* __restrict__ input, value_t* __restrict__ output, std::size_t elements_per_plane)
{
  const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

  value_t acc = 0;

  for (std::size_t idx = tid; idx < elements_per_plane; idx += stride)
  {
    value_t values[LoadsPerThread];

#pragma unroll
    for (int load = 0; load < LoadsPerThread; ++load)
    {
      values[load] = input[idx + static_cast<std::size_t>(load) * elements_per_plane];
    }

#pragma unroll
    for (int load = 0; load < LoadsPerThread; ++load)
    {
      acc += values[load];
    }
  }

  output[tid] = acc;
}

nvbench::summary& add_summary(
  nvbench::state& state, std::string_view tag, std::string_view name, std::string_view description, bool hide = false)
{
  auto& summary = state.add_summary(std::string{tag});
  summary.set_string("name", std::string{name});
  summary.set_string("description", std::string{description});
  if (hide)
  {
    summary.set_string("hide", "");
  }
  return summary;
}

template <int LoadsPerThread, int BlockThreads>
void run_bandwidth_saturation_case(nvbench::state& state)
{
  const auto& device = state.get_device().value();
  const auto memory  = device.get_global_memory_usage();

  const std::size_t desired_input_bytes = std::min(
    max_input_bytes, std::max<std::size_t>(256 * mib, 16 * std::max<std::size_t>(1, device.get_l2_cache_size())));
  const std::size_t available_input_bytes = memory.bytes_free / 2;
  const std::size_t input_bytes_budget    = std::min(desired_input_bytes, available_input_bytes);

  if (input_bytes_budget < min_input_bytes)
  {
    state.skip("Skipping: not enough free device memory for a streaming working set.");
    return;
  }

  int active_blocks_per_sm = 0;
  const auto kernel        = streaming_read_kernel<LoadsPerThread, BlockThreads>;
  NVBENCH_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, BlockThreads, 0));

  if (active_blocks_per_sm <= 0)
  {
    state.skip("Skipping: kernel cannot achieve any active blocks per SM.");
    return;
  }

  const int sm_count                = device.get_number_of_sms();
  const int grid_blocks             = active_blocks_per_sm * sm_count;
  const std::size_t threads_in_grid = static_cast<std::size_t>(grid_blocks) * BlockThreads;

  const std::size_t input_elements      = input_bytes_budget / sizeof(value_t);
  const std::size_t elements_per_plane  = input_elements / LoadsPerThread;
  const std::size_t aligned_input_bytes = elements_per_plane * LoadsPerThread * sizeof(value_t);
  const std::size_t bytes_in_flight_per_sm =
    static_cast<std::size_t>(LoadsPerThread) * sizeof(value_t) * BlockThreads * active_blocks_per_sm;

  if (elements_per_plane == 0)
  {
    state.skip("Skipping: working set is too small for the selected loads-per-thread configuration.");
    return;
  }

  auto& bif_summary = add_summary(
    state,
    "cccl/bandwidth_saturation/bytes_in_flight_per_sm",
    "BIF/SM",
    "Requested bytes in flight per SM: LoadsPerThread * sizeof(value_t) * ThreadsPerBlock * ActiveBlocksPerSM.");
  bif_summary.set_string("hint", "bytes");
  bif_summary.set_int64("value", static_cast<nvbench::int64_t>(bytes_in_flight_per_sm));

  auto& blocks_summary = add_summary(
    state,
    "cccl/bandwidth_saturation/active_blocks_per_sm",
    "Blocks/SM",
    "Max active blocks per SM for the compiled kernel instance.",
    true);
  blocks_summary.set_int64("value", active_blocks_per_sm);

  auto& input_summary = add_summary(
    state,
    "cccl/bandwidth_saturation/input_bytes",
    "Input Bytes",
    "Total streamed input bytes per kernel launch.",
    true);
  input_summary.set_string("hint", "bytes");
  input_summary.set_int64("value", static_cast<nvbench::int64_t>(aligned_input_bytes));

  auto& l2_summary =
    add_summary(state, "cccl/bandwidth_saturation/l2_bytes", "L2 Bytes", "Device L2 cache size in bytes.", true);
  l2_summary.set_string("hint", "bytes");
  l2_summary.set_int64("value", static_cast<nvbench::int64_t>(device.get_l2_cache_size()));

  auto& grid_summary = add_summary(
    state,
    "cccl/bandwidth_saturation/grid_blocks",
    "Grid Blocks",
    "Grid size chosen from max active blocks per SM times SM count.",
    true);
  grid_summary.set_int64("value", grid_blocks);

  try
  {
    thrust::device_vector<value_t> input(elements_per_plane * LoadsPerThread, value_t{1});
    thrust::device_vector<value_t> output(threads_in_grid, thrust::no_init);

    state.add_global_memory_reads(aligned_input_bytes);

    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_blocks, BlockThreads, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), elements_per_plane);
      NVBENCH_CUDA_CALL(cudaPeekAtLastError());
    });
  }
  catch (const std::bad_alloc&)
  {
    state.skip("Skipping: out of memory.");
  }
}

void bandwidth_saturation(nvbench::state& state)
{
  const int loads_per_thread  = static_cast<int>(state.get_int64("LoadsPerThread"));
  const int threads_per_block = static_cast<int>(state.get_int64("ThreadsPerBlock"));

  switch (threads_per_block)
  {
    case 128:
      switch (loads_per_thread)
      {
        case 1:
          return run_bandwidth_saturation_case<1, 128>(state);
        case 2:
          return run_bandwidth_saturation_case<2, 128>(state);
        case 4:
          return run_bandwidth_saturation_case<4, 128>(state);
        case 8:
          return run_bandwidth_saturation_case<8, 128>(state);
        case 12:
          return run_bandwidth_saturation_case<12, 128>(state);
        case 16:
          return run_bandwidth_saturation_case<16, 128>(state);
        case 24:
          return run_bandwidth_saturation_case<24, 128>(state);
        case 32:
          return run_bandwidth_saturation_case<32, 128>(state);
      }
      break;

    case 256:
      switch (loads_per_thread)
      {
        case 1:
          return run_bandwidth_saturation_case<1, 256>(state);
        case 2:
          return run_bandwidth_saturation_case<2, 256>(state);
        case 4:
          return run_bandwidth_saturation_case<4, 256>(state);
        case 8:
          return run_bandwidth_saturation_case<8, 256>(state);
        case 12:
          return run_bandwidth_saturation_case<12, 256>(state);
        case 16:
          return run_bandwidth_saturation_case<16, 256>(state);
        case 24:
          return run_bandwidth_saturation_case<24, 256>(state);
        case 32:
          return run_bandwidth_saturation_case<32, 256>(state);
      }
      break;

    case 512:
      switch (loads_per_thread)
      {
        case 1:
          return run_bandwidth_saturation_case<1, 512>(state);
        case 2:
          return run_bandwidth_saturation_case<2, 512>(state);
        case 4:
          return run_bandwidth_saturation_case<4, 512>(state);
        case 8:
          return run_bandwidth_saturation_case<8, 512>(state);
        case 12:
          return run_bandwidth_saturation_case<12, 512>(state);
        case 16:
          return run_bandwidth_saturation_case<16, 512>(state);
        case 24:
          return run_bandwidth_saturation_case<24, 512>(state);
        case 32:
          return run_bandwidth_saturation_case<32, 512>(state);
      }
      break;
  }

  state.skip("Unsupported LoadsPerThread or ThreadsPerBlock axis value.");
}
} // namespace

NVBENCH_BENCH(bandwidth_saturation)
  .set_name("base")
  .add_int64_axis("LoadsPerThread", {1, 2, 4, 8, 12, 16, 24, 32})
  .add_int64_axis("ThreadsPerBlock", {128, 256, 512});
