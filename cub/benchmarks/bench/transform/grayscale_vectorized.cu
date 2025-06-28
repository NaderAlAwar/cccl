// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:0:0
// %RANGE% TUNE_LOAD_STORE_WORD_SIZE lsws 2:16:2
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 2:8:2

#include "common.h"

// This benchmark tests overlapping memory regions for reading and is compute intensive

using current_offset_types = nvbench::type_list<int32_t>;

struct rgb_t
{
  float r;
  float g;
  float b;

  __host__ __device__ float greyscale() const
  {
    // Convert RGB to greyscale using the luminosity method
    return 0.2989f * r + 0.587f * g + 0.114f * b;
  }
};

struct transform_op_t
{
  __host__ __device__ float operator()(rgb_t pixel) const
  {
    return pixel.greyscale();
  }
};

template <typename OffsetT>
static void grayscale(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<rgb_t> in1(n);
  thrust::device_vector<float> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<rgb_t>(n);
  state.add_global_memory_writes<float>(n);

  bench_transform(state, ::cuda::std::tuple{in1.begin()}, out.begin(), n, transform_op_t{});
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(grayscale, NVBENCH_TYPE_AXES(current_offset_types))
  .set_name("grayscale")
  .add_int64_axis("Elements{io}", {268435456});
