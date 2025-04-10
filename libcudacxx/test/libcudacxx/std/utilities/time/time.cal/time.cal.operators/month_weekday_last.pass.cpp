//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_weekday_last;

// constexpr month_weekday_last
//   operator/(const month& m, const weekday_last& wdl) noexcept;
// Returns: {m, wdl}.
//
// constexpr month_weekday_last
//   operator/(int m, const weekday_last& wdl) noexcept;
// Returns: month(m) / wdl.
//
// constexpr month_weekday_last
//   operator/(const weekday_last& wdl, const month& m) noexcept;
// Returns: m / wdl.
//
// constexpr month_weekday_last
//   operator/(const weekday_last& wdl, int m) noexcept;
// Returns: month(m) / wdl.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month_weekday      = cuda::std::chrono::month_weekday;
  using month              = cuda::std::chrono::month;
  using weekday            = cuda::std::chrono::weekday;
  using weekday_last       = cuda::std::chrono::weekday_last;
  using month_weekday_last = cuda::std::chrono::month_weekday_last;

  constexpr weekday Tuesday                   = cuda::std::chrono::Tuesday;
  constexpr month February                    = cuda::std::chrono::February;
  constexpr cuda::std::chrono::last_spec last = cuda::std::chrono::last;

  { // operator/(const month& m, const weekday_last& wdi) (and switched)
    static_assert(noexcept(February / Tuesday[last]));
    static_assert(cuda::std::is_same_v<month_weekday_last, decltype(February / Tuesday[last])>);
    static_assert(noexcept(Tuesday[last] / February));
    static_assert(cuda::std::is_same_v<month_weekday_last, decltype(Tuesday[last] / February)>);

    //  Run the example
    {
      constexpr month_weekday_last wdi = February / Tuesday[last];
      static_assert(wdi.month() == February, "");
      static_assert(wdi.weekday_last() == Tuesday[last], "");
    }

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 6; ++j)
      {
        month m(i);
        weekday_last wdi        = weekday{j}[last];
        month_weekday_last mwd1 = m / wdi;
        month_weekday_last mwd2 = wdi / m;
        assert(mwd1.month() == m);
        assert(mwd1.weekday_last() == wdi);
        assert(mwd2.month() == m);
        assert(mwd2.weekday_last() == wdi);
        assert(mwd1 == mwd2);
      }
    }
  }

  { // operator/(int m, const weekday_last& wdi) (and switched)
    static_assert(noexcept(2 / Tuesday[2]));
    static_assert(cuda::std::is_same_v<month_weekday_last, decltype(2 / Tuesday[last])>);
    static_assert(noexcept(Tuesday[2] / 2));
    static_assert(cuda::std::is_same_v<month_weekday_last, decltype(Tuesday[last] / 2)>);

    //  Run the example
    {
      constexpr month_weekday wdi = 2 / Tuesday[3];
      static_assert(wdi.month() == February, "");
      static_assert(wdi.weekday_indexed() == Tuesday[3], "");
    }

    for (int i = 1; i <= 12; ++i)
    {
      for (unsigned j = 0; j <= 6; ++j)
      {
        weekday_last wdi        = weekday{j}[last];
        month_weekday_last mwd1 = i / wdi;
        month_weekday_last mwd2 = wdi / i;
        assert(mwd1.month() == month(i));
        assert(mwd1.weekday_last() == wdi);
        assert(mwd2.month() == month(i));
        assert(mwd2.weekday_last() == wdi);
        assert(mwd1 == mwd2);
      }
    }
  }

  return 0;
}
