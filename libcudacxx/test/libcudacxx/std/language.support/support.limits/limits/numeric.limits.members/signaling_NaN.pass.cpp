//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// signaling_NaN()

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "common.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_impl(cuda::std::true_type)
{
  assert(cuda::std::isnan(cuda::std::numeric_limits<T>::signaling_NaN()));
  assert(cuda::std::isnan(cuda::std::numeric_limits<const T>::signaling_NaN()));
  assert(cuda::std::isnan(cuda::std::numeric_limits<volatile T>::signaling_NaN()));
  assert(cuda::std::isnan(cuda::std::numeric_limits<const volatile T>::signaling_NaN()));
}

template <class T>
__host__ __device__ void test_impl(cuda::std::false_type)
{
  assert(float_eq(cuda::std::numeric_limits<T>::signaling_NaN(), T()));
  assert(float_eq(cuda::std::numeric_limits<const T>::signaling_NaN(), T()));
  assert(float_eq(cuda::std::numeric_limits<volatile T>::signaling_NaN(), T()));
  assert(float_eq(cuda::std::numeric_limits<const volatile T>::signaling_NaN(), T()));
}

template <class T>
__host__ __device__ inline void test()
{
  test_impl<T>(cuda::std::integral_constant<bool, cuda::std::numeric_limits<T>::has_signaling_NaN>{});
}

int main(int, char**)
{
  test<bool>();
  test<char>();
  test<signed char>();
  test<unsigned char>();
  test<wchar_t>();
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();
  test<short>();
  test<unsigned short>();
  test<int>();
  test<unsigned int>();
  test<long>();
  test<unsigned long>();
  test<long long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
