//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday;

// constexpr chrono::year year() const noexcept;
//  Returns: d_

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using month              = cuda::std::chrono::month;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;

  static_assert(noexcept(cuda::std::declval<const year_month_weekday>().year()));
  static_assert(cuda::std::is_same_v<year, decltype(cuda::std::declval<const year_month_weekday>().year())>);

  static_assert(year_month_weekday{}.year() == year{}, "");

  for (int i = 1; i <= 50; ++i)
  {
    year_month_weekday ym(year{i}, month{1}, weekday_indexed{});
    assert(static_cast<int>(ym.year()) == i);
  }

  return 0;
}
