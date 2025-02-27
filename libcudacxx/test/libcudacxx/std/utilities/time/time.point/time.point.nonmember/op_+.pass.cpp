//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class Clock, class Duration1, class Rep2, class Period2>
//   time_point<Clock, typename common_type<Duration1, duration<Rep2, Period2>>::type>
//   operator+(const time_point<Clock, Duration1>& lhs, const duration<Rep2, Period2>& rhs);

// template <class Rep1, class Period1, class Clock, class Duration2>
//   time_point<Clock, typename common_type<duration<Rep1, Period1>, Duration2>::type>
//   operator+(const duration<Rep1, Period1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock Clock;
  typedef cuda::std::chrono::milliseconds Duration1;
  typedef cuda::std::chrono::microseconds Duration2;
  {
    cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    cuda::std::chrono::time_point<Clock, Duration2> t2 = t1 + Duration2(5);
    assert(t2.time_since_epoch() == Duration2(3005));
    t2 = Duration2(6) + t1;
    assert(t2.time_since_epoch() == Duration2(3006));
  }
  {
    constexpr cuda::std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    constexpr cuda::std::chrono::time_point<Clock, Duration2> t2 = t1 + Duration2(5);
    static_assert(t2.time_since_epoch() == Duration2(3005), "");
    constexpr cuda::std::chrono::time_point<Clock, Duration2> t3 = Duration2(6) + t1;
    static_assert(t3.time_since_epoch() == Duration2(3006), "");
  }

  return 0;
}
