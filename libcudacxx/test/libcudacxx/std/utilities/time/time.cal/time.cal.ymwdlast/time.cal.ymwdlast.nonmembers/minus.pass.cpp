//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday_last;

// constexpr year_month_weekday_last operator-(const year_month_weekday_last& ymwdl, const months& dm) noexcept;
//   Returns: ymwdl + (-dm).
//
// constexpr year_month_weekday_last operator-(const year_month_weekday_last& ymwdl, const years& dy) noexcept;
//   Returns: ymwdl + (-dy).

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool testConstexprYears(cuda::std::chrono::year_month_weekday_last ym)
{
  cuda::std::chrono::years offset{14};
  if (static_cast<int>((ym).year()) != 66)
  {
    return false;
  }
  if (static_cast<int>((ym - offset).year()) != 52)
  {
    return false;
  }
  return true;
}

__host__ __device__ constexpr bool testConstexprMonths(cuda::std::chrono::year_month_weekday_last ym)
{
  cuda::std::chrono::months offset{6};
  if (static_cast<unsigned>((ym).month()) != 10)
  {
    return false;
  }
  if (static_cast<unsigned>((ym - offset).month()) != 4)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;
  using years                   = cuda::std::chrono::years;
  using months                  = cuda::std::chrono::months;

  constexpr month October   = cuda::std::chrono::October;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  { // year_month_weekday_last - years

    static_assert(noexcept(cuda::std::declval<year_month_weekday_last>() - cuda::std::declval<years>()));
    static_assert(
      cuda::std::is_same_v<year_month_weekday_last,
                           decltype(cuda::std::declval<year_month_weekday_last>() - cuda::std::declval<years>())>);

    static_assert(testConstexprYears(year_month_weekday_last{year{66}, October, weekday_last{Tuesday}}), "");

    year_month_weekday_last ym{year{1234}, October, weekday_last{Tuesday}};
    for (int i = 0; i <= 10; ++i)
    {
      year_month_weekday_last ym1 = ym - years{i};
      assert(ym1.year() == year{1234 - i});
      assert(ym1.month() == October);
      assert(ym1.weekday() == Tuesday);
      assert(ym1.weekday_last() == weekday_last{Tuesday});
    }
  }

  { // year_month_weekday_last - months

    static_assert(noexcept(cuda::std::declval<year_month_weekday_last>() - cuda::std::declval<months>()));
    static_assert(
      cuda::std::is_same_v<year_month_weekday_last,
                           decltype(cuda::std::declval<year_month_weekday_last>() - cuda::std::declval<months>())>);

    static_assert(testConstexprMonths(year_month_weekday_last{year{66}, October, weekday_last{Tuesday}}), "");

    year_month_weekday_last ym{year{1234}, October, weekday_last{Tuesday}};
    for (unsigned i = 0; i < 10; ++i)
    {
      year_month_weekday_last ym1 = ym - months{i};
      assert(ym1.year() == year{1234});
      assert(ym1.month() == month{10 - i});
      assert(ym1.weekday() == Tuesday);
      assert(ym1.weekday_last() == weekday_last{Tuesday});
    }
  }

  return 0;
}
