//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr bool operator==(const year& x, const year& y) noexcept;
// constexpr bool operator!=(const year& x, const year& y) noexcept;
// constexpr bool operator< (const year& x, const year& y) noexcept;
// constexpr bool operator> (const year& x, const year& y) noexcept;
// constexpr bool operator<=(const year& x, const year& y) noexcept;
// constexpr bool operator>=(const year& x, const year& y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  AssertComparisonsAreNoexcept<year>();
  AssertComparisonsReturnBool<year>();

  static_assert(testComparisonsValues<year>(0, 0), "");
  static_assert(testComparisonsValues<year>(0, 1), "");

  //  Some 'ok' values as well
  static_assert(testComparisonsValues<year>(5, 5), "");
  static_assert(testComparisonsValues<year>(5, 10), "");

  for (int i = 1; i < 10; ++i)
  {
    for (int j = 1; j < 10; ++j)
    {
      assert(testComparisonsValues<year>(i, j));
    }
  }

  return 0;
}
