//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday_last;

// constexpr bool ok() const noexcept;
//  Returns: y_.ok() && m_.ok() && wdl_.ok().

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;

  constexpr month January   = cuda::std::chrono::January;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(noexcept(cuda::std::declval<const year_month_weekday_last>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const year_month_weekday_last>().ok())>);

  static_assert(!year_month_weekday_last{year{-32768}, month{}, weekday_last{weekday{}}}.ok(), ""); // All three bad

  static_assert(!year_month_weekday_last{year{-32768}, January, weekday_last{Tuesday}}.ok(), ""); // Bad year
  static_assert(!year_month_weekday_last{year{2019}, month{}, weekday_last{Tuesday}}.ok(), ""); // Bad month
  static_assert(!year_month_weekday_last{year{2019}, January, weekday_last{weekday{8}}}.ok(), ""); // Bad day

  static_assert(!year_month_weekday_last{year{-32768}, month{}, weekday_last{Tuesday}}.ok(), ""); // Bad year & month
  static_assert(!year_month_weekday_last{year{2019}, month{}, weekday_last{weekday{8}}}.ok(), ""); // Bad month & day
  static_assert(!year_month_weekday_last{year{-32768}, January, weekday_last{weekday{8}}}.ok(), ""); // Bad year & day

  static_assert(year_month_weekday_last{year{2019}, January, weekday_last{Tuesday}}.ok(), ""); // All OK

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_weekday_last ym{year{2019}, January, weekday_last{Tuesday}};
    assert((ym.ok() == weekday_last{Tuesday}.ok()));
  }

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_weekday_last ym{year{2019}, January, weekday_last{weekday{i}}};
    assert((ym.ok() == weekday_last{weekday{i}}.ok()));
  }

  for (unsigned i = 0; i <= 50; ++i)
  {
    year_month_weekday_last ym{year{2019}, month{i}, weekday_last{Tuesday}};
    assert((ym.ok() == month{i}.ok()));
  }

  const int ymax = static_cast<int>(year::max());
  for (int i = ymax - 100; i <= ymax + 100; ++i)
  {
    year_month_weekday_last ym{year{i}, January, weekday_last{Tuesday}};
    assert((ym.ok() == year{i}.ok()));
  }

  return 0;
}
