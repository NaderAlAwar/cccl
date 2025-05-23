//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   requires OutputIterator<Iter, RvalueOf<Iter::reference>::type>
//         && HasEqualTo<Iter::value_type, T>
//   constexpr Iter         // constexpr after C++17
//   remove(Iter first, Iter last, const T& value);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct MoveOnly
{
  int val_;

  __host__ __device__ constexpr MoveOnly(const int val) noexcept
      : val_(val)
  {}

  MoveOnly()                           = default;
  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly(MoveOnly&&)                 = default;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly& operator=(MoveOnly&&)      = default;

  __host__ __device__ constexpr bool operator==(const int val) const noexcept
  {
    return val_ == val;
  }

  __host__ __device__ constexpr bool operator==(const MoveOnly& other) const noexcept
  {
    return val_ == other.val_;
  }
};

template <class Iter>
constexpr __host__ __device__ void test()
{
  using value_type = typename cuda::std::iterator_traits<Iter>::value_type;

  constexpr int N                      = 9;
  value_type ia[N]                     = {0, 1, 2, 3, 4, 2, 3, 4, 2};
  constexpr value_type expected[N - 3] = {0, 1, 3, 4, 3, 4};
  Iter r                               = cuda::std::remove(Iter(ia), Iter(ia + N), 2);
  assert(base(r) == ia + N - 3);
  for (int i = 0; i < N - 3; ++i)
  {
    assert(ia[i] == expected[i]);
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  test<cpp17_input_iterator<MoveOnly*>>();
  test<forward_iterator<MoveOnly*>>();
  test<bidirectional_iterator<MoveOnly*>>();
  test<random_access_iterator<MoveOnly*>>();
  test<MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
