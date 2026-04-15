/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <cstdint>

#include <benchmarks/common/cudf_random_input.cuh>

template std::unique_ptr<cudf::column> cub::benchmarks::cudf_input::make_random_fixed_width_column<std::int16_t>(
  cudf::size_type, unsigned, bool, unsigned, double);
template std::unique_ptr<cudf::column> cub::benchmarks::cudf_input::make_random_fixed_width_column<std::uint32_t>(
  cudf::size_type, unsigned, bool, unsigned, double);
template std::unique_ptr<cudf::column>
cub::benchmarks::cudf_input::make_random_fixed_width_column<double>(cudf::size_type, unsigned, bool, unsigned, double);
template std::unique_ptr<cudf::column>
cub::benchmarks::cudf_input::make_random_fixed_width_column<bool>(cudf::size_type, unsigned, bool, unsigned, double);

template std::unique_ptr<cudf::table>
cub::benchmarks::cudf_input::make_copy_if_else_input<std::int16_t>(cudf::size_type, bool);
template std::unique_ptr<cudf::table>
cub::benchmarks::cudf_input::make_copy_if_else_input<std::uint32_t>(cudf::size_type, bool);
template std::unique_ptr<cudf::table>
cub::benchmarks::cudf_input::make_copy_if_else_input<double>(cudf::size_type, bool);
