//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day_last;

// constexpr chrono::day day() const noexcept;
//  Returns: d_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;

  static_assert(noexcept(cuda::std::declval<const year_month_day_last>().year()));
  static_assert(cuda::std::is_same_v<year, decltype(cuda::std::declval<const year_month_day_last>().year())>);

  for (int i = 1; i <= 50; ++i)
  {
    year_month_day_last ym(year{i}, month_day_last{month{}});
    assert(static_cast<int>(ym.year()) == i);
  }

  return 0;
}
