//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The CI "Apple back-deployment with assertions enabled" needs a higher value
// CONSTEXPR_STEPS: 12712420

// bitset<N> operator<<(size_t pos) const; // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <cuda::std::size_t N, cuda::std::size_t Start = 0, cuda::std::size_t End = static_cast<cuda::std::size_t>(-1)>
__host__ __device__ constexpr bool test_left_shift()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  if (Start == 9)
  {
    assert(End >= cases.size());
  }
  for (cuda::std::size_t c = Start; c != cases.size() && c != End; ++c)
  {
    for (cuda::std::size_t s = 0; s <= N + 1; ++s)
    {
      cuda::std::bitset<N> v1(cases[c]);
      cuda::std::bitset<N> v2 = v1;
      assert((v1 <<= s) == (v2 << s));
    }
  }

  return true;
}

int main(int, char**)
{
  test_left_shift<0>();
  test_left_shift<1>();
  test_left_shift<31>();
  test_left_shift<32>();
  test_left_shift<33>();
  test_left_shift<63>();
  test_left_shift<64>();
  test_left_shift<65>();
  test_left_shift<1000>(); // not in constexpr because of constexpr evaluation step limits
  static_assert(test_left_shift<0>(), "");
  static_assert(test_left_shift<1>(), "");
  static_assert(test_left_shift<31>(), "");
  static_assert(test_left_shift<32>(), "");
  static_assert(test_left_shift<33>(), "");
  static_assert(test_left_shift<63, 0, 6>(), "");
  static_assert(test_left_shift<63, 6>(), "");
  static_assert(test_left_shift<64, 0, 6>(), "");
  static_assert(test_left_shift<64, 6>(), "");
  static_assert(test_left_shift<65, 0, 3>(), "");
  static_assert(test_left_shift<65, 3, 6>(), "");
  static_assert(test_left_shift<65, 6, 9>(), "");
  static_assert(test_left_shift<65, 9>(), "");

  return 0;
}
