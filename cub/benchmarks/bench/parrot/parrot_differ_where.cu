// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/*
 * NVBench benchmark for sushi.differ().where() using parrot C++ library.
 *
 * This benchmark tests the parrot high-level API which internally uses
 * Thrust/CUB iterators and DevicePartition.
 */

// Include parrot from external location
#include <thrust/device_vector.h>

#include <cstdlib>
#include <vector>

#include "/home/coder/parrot-cuda/parrot.hpp"
#include <nvbench/nvbench.cuh>

void parrot_differ_where(nvbench::state& state)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements"));

  // Create random 0s and 1s on host
  std::vector<int> h_data(n);
  std::srand(42); // Fixed seed for reproducibility
  for (std::size_t i = 0; i < n; ++i)
  {
    h_data[i] = std::rand() % 2;
  }

  // Create parrot array from host data
  // auto sushi = parrot::array(h_data);
  auto sushi = parrot::scalar(2).repeat(n).rand();
  // Add metrics
  state.add_element_count(n - 1); // differ reduces length by 1
  state.add_global_memory_reads<int>(n);
  state.add_global_memory_writes<int>((n - 1) / 2); // Approximate output (half selected)

  // Run benchmark
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Run the differ().where() chain and force evaluation with apply()
    // (keeps data on GPU, doesn't copy to host)
    auto result = sushi.differ().where().apply();
  });
}

NVBENCH_BENCH(parrot_differ_where).set_name("parrot_differ_where").add_int64_axis("Elements", {100000000}); // 100M
