//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_enum

#include <cuda/std/cstddef> // for cuda::std::nullptr_t
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_enum()
{
  static_assert(cuda::std::is_enum<T>::value, "");
  static_assert(cuda::std::is_enum<const T>::value, "");
  static_assert(cuda::std::is_enum<volatile T>::value, "");
  static_assert(cuda::std::is_enum<const volatile T>::value, "");
  static_assert(cuda::std::is_enum_v<T>, "");
  static_assert(cuda::std::is_enum_v<const T>, "");
  static_assert(cuda::std::is_enum_v<volatile T>, "");
  static_assert(cuda::std::is_enum_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_enum()
{
  static_assert(!cuda::std::is_enum<T>::value, "");
  static_assert(!cuda::std::is_enum<const T>::value, "");
  static_assert(!cuda::std::is_enum<volatile T>::value, "");
  static_assert(!cuda::std::is_enum<const volatile T>::value, "");
  static_assert(!cuda::std::is_enum_v<T>, "");
  static_assert(!cuda::std::is_enum_v<const T>, "");
  static_assert(!cuda::std::is_enum_v<volatile T>, "");
  static_assert(!cuda::std::is_enum_v<const volatile T>, "");
}

class Empty
{};

class NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  __host__ __device__ virtual ~Abstract() = 0;
};

enum Enum
{
  zero,
  one
};
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
  test_is_enum<Enum>();

  test_is_not_enum<cuda::std::nullptr_t>();
  test_is_not_enum<void>();
  test_is_not_enum<int>();
  test_is_not_enum<int&>();
  test_is_not_enum<int&&>();
  test_is_not_enum<int*>();
  test_is_not_enum<double>();
  test_is_not_enum<const int*>();
  test_is_not_enum<char[3]>();
  test_is_not_enum<char[]>();
  test_is_not_enum<Union>();
  test_is_not_enum<Empty>();
  test_is_not_enum<bit_zero>();
  test_is_not_enum<NotEmpty>();
  test_is_not_enum<Abstract>();
  test_is_not_enum<FunctionPtr>();
  test_is_not_enum<incomplete_type>();

  return 0;
}
