//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day_last;

// constexpr operator local_days() const noexcept;
//  Returns: local_days{sys_days{*this}.time_since_epoch()}.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using local_days          = cuda::std::chrono::local_days;
  using days                = cuda::std::chrono::days;

  static_assert(noexcept(static_cast<local_days>(cuda::std::declval<const year_month_day_last>())));
  static_assert(
    cuda::std::is_same_v<local_days, decltype(static_cast<local_days>(cuda::std::declval<const year_month_day_last>()))>);

  { // Last day in Jan 1970 was the 31st
    constexpr year_month_day_last ymdl{year{1970}, month_day_last{cuda::std::chrono::January}};
    constexpr local_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{30}, "");
  }

  {
    constexpr year_month_day_last ymdl{year{2000}, month_day_last{cuda::std::chrono::January}};
    constexpr local_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{10957 + 30}, "");
  }

  {
    constexpr year_month_day_last ymdl{year{1940}, month_day_last{cuda::std::chrono::January}};
    constexpr local_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{-10957 + 29}, "");
  }

  {
    year_month_day_last ymdl{year{1939}, month_day_last{cuda::std::chrono::November}};
    local_days sd{ymdl};

    assert(sd.time_since_epoch() == days{-(10957 + 33)});
  }

  return 0;
}
