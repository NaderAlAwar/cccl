//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class G>
//   constexpr explicit(!is_convertible_v<const G&, E>) expected(const unexpected<G>& e);
//
// Let GF be const G&
//
// Constraints: is_constructible_v<E, GF> is true.
//
// Effects: Direct-non-list-initializes unex with cuda::std::forward<GF>(e.error()).
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints
static_assert(cuda::std::is_constructible_v<cuda::std::expected<int, int>, const cuda::std::unexpected<int>&>, "");

// !is_constructible_v<E, GF>
struct foo
{};
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<int, int>, const cuda::std::unexpected<foo>&>, "");
static_assert(
  !cuda::std::is_constructible_v<cuda::std::expected<int, MoveOnly>, const cuda::std::unexpected<MoveOnly>&>, "");

// explicit(!is_convertible_v<const G&, E>)
struct NotConvertible
{
  __host__ __device__ explicit NotConvertible(int);
};
static_assert(cuda::std::is_convertible_v<const cuda::std::unexpected<int>&, cuda::std::expected<int, int>>, "");
static_assert(!cuda::std::is_convertible_v<const cuda::std::unexpected<int>&, cuda::std::expected<int, NotConvertible>>,
              "");

struct MyInt
{
  int i;
  __host__ __device__ constexpr MyInt(int ii)
      : i(ii)
  {}
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i == rhs.i;
  };
  __host__ __device__ friend constexpr bool operator!=(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i != rhs.i;
  };
#endif // TEST_STD_VER > 2017
};

template <class T>
__host__ __device__ constexpr void testUnexpected()
{
  const cuda::std::unexpected<int> u(5);
  cuda::std::expected<int, T> e(u);
  assert(!e.has_value());
  assert(e.error() == 5);
}

__host__ __device__ constexpr bool test()
{
  testUnexpected<int>();
  testUnexpected<MyInt>();
  return true;
}

#if TEST_HAS_EXCEPTIONS()
struct Except
{};
struct Throwing
{
  Throwing(int)
  {
    throw Except{};
  }
};
void test_exceptions()
{
  const cuda::std::unexpected<int> u(5);
  try
  {
    cuda::std::expected<int, Throwing> e(u);
    unused(e);
    assert(false);
  }
  catch (const Except&)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
