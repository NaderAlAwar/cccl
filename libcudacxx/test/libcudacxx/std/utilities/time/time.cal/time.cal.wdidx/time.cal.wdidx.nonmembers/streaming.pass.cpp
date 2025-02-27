//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

// <chrono>
// class weekday_indexed;

// template<class charT, class traits>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os, const weekday_indexed& wdi);
//
//   Effects: os << wdi.weekday() << '[' << wdi.index().
//     If wdi.index() is in the range [1, 5], appends with ']',
//       otherwise appends with " is not a valid index]".

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday_indexed = cuda::std::chrono::weekday_indexed;
  using weekday         = cuda::std::chrono::weekday;

  std::cout << weekday_indexed{weekday{3}};

  return 0;
}
