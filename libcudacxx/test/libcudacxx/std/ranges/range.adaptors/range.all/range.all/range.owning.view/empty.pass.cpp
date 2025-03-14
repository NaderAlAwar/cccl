//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr bool empty() requires requires { ranges::empty(r_); }
// constexpr bool empty() const requires requires { ranges::empty(r_); }

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
_CCCL_CONCEPT HasEmpty = _CCCL_REQUIRES_EXPR((T), T t)(unused(t.empty()));

struct ComparableIters
{
  __host__ __device__ forward_iterator<int*> begin();
  __host__ __device__ forward_iterator<int*> end();
};

struct NoEmpty
{
  __host__ __device__ cpp20_input_iterator<int*> begin();
  __host__ __device__ sentinel_wrapper<cpp20_input_iterator<int*>> end();
};

struct EmptyMember
{
  __host__ __device__ cpp20_input_iterator<int*> begin();
  __host__ __device__ sentinel_wrapper<cpp20_input_iterator<int*>> end();
  __host__ __device__ bool empty() const;
};

__host__ __device__ constexpr bool test()
{
  {
    using OwningView = cuda::std::ranges::owning_view<ComparableIters>;
    static_assert(HasEmpty<OwningView&>);
    static_assert(HasEmpty<OwningView&&>);
    static_assert(!HasEmpty<const OwningView&>);
    static_assert(!HasEmpty<const OwningView&&>);
  }
  {
    static_assert(cuda::std::ranges::range<NoEmpty&>);
    static_assert(!cuda::std::invocable<decltype(cuda::std::ranges::empty), NoEmpty&>);
    static_assert(!cuda::std::ranges::range<const NoEmpty&>); // no begin/end
    static_assert(!cuda::std::invocable<decltype(cuda::std::ranges::empty), const NoEmpty&>);
    using OwningView = cuda::std::ranges::owning_view<NoEmpty>;
    static_assert(!HasEmpty<OwningView&>);
    static_assert(!HasEmpty<OwningView&&>);
    static_assert(!HasEmpty<const OwningView&>);
    static_assert(!HasEmpty<const OwningView&&>);
  }
  {
    static_assert(cuda::std::ranges::range<EmptyMember&>);
    static_assert(cuda::std::invocable<decltype(cuda::std::ranges::empty), EmptyMember&>);
    static_assert(!cuda::std::ranges::range<const EmptyMember&>); // no begin/end
    static_assert(cuda::std::invocable<decltype(cuda::std::ranges::empty), const EmptyMember&>);
    using OwningView = cuda::std::ranges::owning_view<EmptyMember>;
    static_assert(cuda::std::ranges::range<OwningView&>);
    static_assert(!cuda::std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasEmpty<OwningView&>);
    static_assert(HasEmpty<OwningView&&>);
    static_assert(HasEmpty<const OwningView&>); // but it still has empty()
    static_assert(HasEmpty<const OwningView&&>);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a));
    assert(ov.empty());
    assert(cuda::std::as_const(ov).empty());
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a + 1));
    assert(!ov.empty());
    assert(!cuda::std::as_const(ov).empty());
  }
  {
    // Test a non-view.
    cuda::std::array<int, 2> a = {1, 2};
    auto ov                    = cuda::std::ranges::owning_view(cuda::std::move(a));
    assert(!ov.empty());
    assert(!cuda::std::as_const(ov).empty());
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
