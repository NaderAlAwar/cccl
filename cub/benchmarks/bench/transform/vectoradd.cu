// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

#include "common.h"

// This benchmark tests overlapping memory regions for reading and is compute intensive

using current_offset_types = nvbench::type_list<int32_t>;

template <typename OffsetT>
static void vectoradd(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<__half> in1(n, startA);
  thrust::device_vector<__half> in2(n, startB);
  thrust::device_vector<__half> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<__half>(n);
  state.add_global_memory_reads<__half>(n);
  state.add_global_memory_writes<__half>(n);

  // the complex comparison needs lots of compute and transform reads from overlapping input
  using compare_op = less_t;
  bench_transform(state, ::cuda::std::tuple{in1.begin(), in2.begin()}, out.begin(), n, ::cuda::std::plus<>{});
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(vectoradd, NVBENCH_TYPE_AXES(current_offset_types))
  .set_name("vectoradd")
  .add_int64_axis("Elements{io}", {268435456});
