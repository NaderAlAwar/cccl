//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// <cuda/std/iterator>
//
// reverse_iterator
//
// pointer operator->() const;

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"

template <class T>
concept HasArrow = requires(T t) { t.operator->(); };

struct simple_bidirectional_iterator
{
  using iterator_category = cuda::std::bidirectional_iterator_tag;
  using value_type        = int;
  using difference_type   = int;
  using pointer           = int*;
  using reference         = int&;

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;

  __host__ __device__ simple_bidirectional_iterator& operator++();
  __host__ __device__ simple_bidirectional_iterator& operator--();
  __host__ __device__ simple_bidirectional_iterator operator++(int);
  __host__ __device__ simple_bidirectional_iterator operator--(int);

  __host__ __device__ friend bool operator==(const simple_bidirectional_iterator&, const simple_bidirectional_iterator&);
};
static_assert(cuda::std::bidirectional_iterator<simple_bidirectional_iterator>);
static_assert(!cuda::std::random_access_iterator<simple_bidirectional_iterator>);

using PtrRI = cuda::std::reverse_iterator<int*>;
static_assert(HasArrow<PtrRI>);

using PtrLikeRI = cuda::std::reverse_iterator<simple_bidirectional_iterator>;
static_assert(HasArrow<PtrLikeRI>);

// `bidirectional_iterator` from `test_iterators.h` doesn't define `operator->`.
using NonPtrRI = cuda::std::reverse_iterator<bidirectional_iterator<int*>>;
static_assert(!HasArrow<NonPtrRI>);

int main(int, char**)
{
  return 0;
}
