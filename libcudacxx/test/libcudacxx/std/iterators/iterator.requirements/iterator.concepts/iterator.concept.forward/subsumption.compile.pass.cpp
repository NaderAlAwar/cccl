//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17
// XFAIL: nvcc-12.0 || nvcc-12.1 || nvcc-12.2 || nvcc-12.3
// nvbug 3885350

// cuda::std::forward_iterator;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

// clang-format off
template<cuda::std::input_iterator>
__host__ __device__ constexpr bool check_subsumption() {
  return false;
}

template<cuda::std::forward_iterator>
__host__ __device__ constexpr bool check_subsumption() {
  return true;
}
// clang-format on

static_assert(check_subsumption<int*>(), "");

int main(int, char**)
{
  return 0;
}
