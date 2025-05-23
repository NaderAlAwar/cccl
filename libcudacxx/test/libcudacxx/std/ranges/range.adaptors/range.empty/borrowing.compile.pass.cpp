//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<class T>
//   inline constexpr bool enable_borrowed_range<empty_view<T>> = true;

#include <cuda/std/ranges>

#include "test_range.h"

static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::empty_view<int>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::empty_view<int*>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::empty_view<BorrowedView>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::empty_view<NonBorrowedView>>);

int main(int, char**)
{
  return 0;
}
