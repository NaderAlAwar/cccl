//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N> operator|(const bitset<N>& lhs, const bitset<N>& rhs); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <cuda::std::size_t N>
__host__ __device__ constexpr bool test_op_or()
{
  auto const& cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c1 = 0; c1 != cases.size(); ++c1)
  {
    for (cuda::std::size_t c2 = 0; c2 != cases.size(); ++c2)
    {
      cuda::std::bitset<N> v1(cases[c1]);
      cuda::std::bitset<N> v2(cases[c2]);
      cuda::std::bitset<N> v3 = v1;
      assert((v1 | v2) == (v3 |= v2));
    }
  }

  return true;
}

int main(int, char**)
{
  test_op_or<0>();
  test_op_or<1>();
  test_op_or<31>();
  test_op_or<32>();
  test_op_or<33>();
  test_op_or<63>();
  test_op_or<64>();
  test_op_or<65>();
  test_op_or<1000>(); // not in constexpr because of constexpr evaluation step limits
  static_assert(test_op_or<0>(), "");
  static_assert(test_op_or<1>(), "");
  static_assert(test_op_or<31>(), "");
  static_assert(test_op_or<32>(), "");
  static_assert(test_op_or<33>(), "");
  static_assert(test_op_or<63>(), "");
  static_assert(test_op_or<64>(), "");
  static_assert(test_op_or<65>(), "");

  return 0;
}
