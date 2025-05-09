//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

// template<size_t Count>
//  constexpr span<element_type, Count> first() const;
//
//  Requires: Count <= size().

#include <cuda/std/cstddef>
#include <cuda/std/span>

void f()
{
  int array[] = {1, 2, 3, 4};
  cuda::std::span<const int, 4> sp(array);

  //  Count too large
  [[maybe_unused]] auto s1 = sp.first<5>(); // expected-error@span:* {{span<T, N>::first<Count>(): Count out of range}}

  //  Count numeric_limits
  [[maybe_unused]] auto s2 = sp.first<cuda::std::size_t(-1)>(); // expected-error@span:* {{span<T, N>::first<Count>():
                                                                // Count out of range}}
}

int main(int, char**)
{
  return 0;
}
