//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// enable_if

#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  ASSERT_SAME_TYPE(void, cuda::std::enable_if<true>::type);
  ASSERT_SAME_TYPE(int, cuda::std::enable_if<true, int>::type);
  ASSERT_SAME_TYPE(void, cuda::std::enable_if_t<true, void>);
  ASSERT_SAME_TYPE(int, cuda::std::enable_if_t<true, int>);

  return 0;
}
