//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_copy_constructible

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_nothrow_copy_constructible()
{
  static_assert(cuda::std::is_nothrow_copy_constructible<T>::value, "");
  static_assert(cuda::std::is_nothrow_copy_constructible<const T>::value, "");
  static_assert(cuda::std::is_nothrow_copy_constructible_v<T>, "");
  static_assert(cuda::std::is_nothrow_copy_constructible_v<const T>, "");
}

template <class T>
__host__ __device__ void test_has_not_nothrow_copy_constructor()
{
  static_assert(!cuda::std::is_nothrow_copy_constructible<T>::value, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible<const T>::value, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible<volatile T>::value, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible<const volatile T>::value, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible_v<T>, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible_v<const T>, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible_v<volatile T>, "");
  static_assert(!cuda::std::is_nothrow_copy_constructible_v<const volatile T>, "");
}

class Empty
{};

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  __host__ __device__ A(const A&);
};

int main(int, char**)
{
  test_has_not_nothrow_copy_constructor<void>();
#if !TEST_COMPILER(NVHPC)
  test_has_not_nothrow_copy_constructor<A>();
#endif // !TEST_COMPILER(NVHPC)

  test_is_nothrow_copy_constructible<int&>();
  test_is_nothrow_copy_constructible<Union>();
  test_is_nothrow_copy_constructible<Empty>();
  test_is_nothrow_copy_constructible<int>();
  test_is_nothrow_copy_constructible<double>();
  test_is_nothrow_copy_constructible<int*>();
  test_is_nothrow_copy_constructible<const int*>();
  test_is_nothrow_copy_constructible<bit_zero>();

  return 0;
}
