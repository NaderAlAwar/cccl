//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_weekday;
//   month_weekday represents the nth weekday of a month, of an as yet unspecified year.

//  constexpr month_weekday(const chrono::month& m, const chrono::weekday_indexed& wdi) noexcept;
//    Effects:  Constructs an object of type month_weekday by initializing m_ with m, and wdi_ with wdi.
//
//  constexpr chrono::month                     month() const noexcept;
//  constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  constexpr bool                                 ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday         = cuda::std::chrono::weekday;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;

  static_assert(noexcept(month_weekday{month{1}, weekday_indexed{weekday{}, 1}}));

  constexpr month_weekday md0{month{}, weekday_indexed{}};
  static_assert(md0.month() == month{}, "");
  static_assert(md0.weekday_indexed() == weekday_indexed{}, "");
  static_assert(!md0.ok(), "");

  constexpr month_weekday md1{cuda::std::chrono::January, weekday_indexed{cuda::std::chrono::Friday, 4}};
  static_assert(md1.month() == cuda::std::chrono::January, "");
  static_assert(md1.weekday_indexed() == weekday_indexed{cuda::std::chrono::Friday, 4}, "");
  static_assert(md1.ok(), "");

  return 0;
}
