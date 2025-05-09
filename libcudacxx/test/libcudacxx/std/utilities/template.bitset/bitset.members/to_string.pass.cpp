//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test:

// template <class charT, class traits, class Allocator>
// basic_string<charT, traits, Allocator>
// to_string(charT zero = charT('0'), charT one = charT('1')) const; // constexpr since C++23
//
// template <class charT, class traits>
// basic_string<charT, traits, allocator<charT> > to_string() const; // constexpr since C++23
//
// template <class charT>
// basic_string<charT, char_traits<charT>, allocator<charT> > to_string() const; // constexpr since C++23
//
// basic_string<char, char_traits<char>, allocator<char> > to_string() const; // constexpr since C++23

#include <cuda/std/version>

#ifndef __LIBCUDACXX_HAS_STRING

int main(int, char**)
{
  return 0;
}

#else

#  include <cuda/std/bitset>
#  include <cuda/std/cassert>
#  include <cuda/std/cstddef>
#  include <cuda/std/memory> // for cuda::std::allocator
#  include <cuda/std/string>
#  include <cuda/std/vector>

#  include "../bitset_test_cases.h"
#  include "test_macros.h"

template <class CharT, cuda::std::size_t N>
constexpr void check_equal(cuda::std::basic_string<CharT> const& s, cuda::std::bitset<N> const& b, CharT zero, CharT one)
{
  assert(s.size() == b.size());
  for (cuda::std::size_t i = 0; i < b.size(); ++i)
  {
    if (b[i])
    {
      assert(s[b.size() - 1 - i] == one);
    }
    else
    {
      assert(s[b.size() - 1 - i] == zero);
    }
  }
}

template <cuda::std::size_t N>
constexpr bool test_to_string()
{
  cuda::std::vector<cuda::std::bitset<N>> const cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> const v = cases[c];
    {
      cuda::std::string s = v.template to_string<char>();
      check_equal(s, v, '0', '1');
    }
    {
      cuda::std::string s = v.to_string();
      check_equal(s, v, '0', '1');
    }
    {
      cuda::std::string s = v.template to_string<char>('0');
      check_equal(s, v, '0', '1');
    }
    {
      cuda::std::string s = v.to_string('0');
      check_equal(s, v, '0', '1');
    }
    {
      cuda::std::string s = v.template to_string<char>('0', '1');
      check_equal(s, v, '0', '1');
    }
    {
      cuda::std::string s = v.to_string('0', '1');
      check_equal(s, v, '0', '1');
    }
    {
      cuda::std::string s = v.to_string('x', 'y');
      check_equal(s, v, 'x', 'y');
    }
  }
  return true;
}

template <cuda::std::size_t N>
constexpr bool test_to_string_wchar()
{
  cuda::std::vector<cuda::std::bitset<N>> const cases = get_test_cases(cuda::std::integral_constant<int, N>());
  for (cuda::std::size_t c = 0; c != cases.size(); ++c)
  {
    cuda::std::bitset<N> const v = cases[c];
    {
      cuda::std::wstring s =
        v.template to_string<wchar_t, cuda::std::char_traits<wchar_t>, cuda::std::allocator<wchar_t>>();
      check_equal(s, v, L'0', L'1');
    }
    {
      cuda::std::wstring s = v.template to_string<wchar_t, cuda::std::char_traits<wchar_t>>();
      check_equal(s, v, L'0', L'1');
    }
    {
      cuda::std::wstring s =
        v.template to_string<wchar_t, cuda::std::char_traits<wchar_t>, cuda::std::allocator<wchar_t>>('0');
      check_equal(s, v, L'0', L'1');
    }
    {
      cuda::std::wstring s = v.template to_string<wchar_t, cuda::std::char_traits<wchar_t>>('0');
      check_equal(s, v, L'0', L'1');
    }
    {
      cuda::std::wstring s =
        v.template to_string<wchar_t, cuda::std::char_traits<wchar_t>, cuda::std::allocator<wchar_t>>('0', '1');
      check_equal(s, v, L'0', L'1');
    }
    {
      cuda::std::wstring s = v.template to_string<wchar_t, cuda::std::char_traits<wchar_t>>('0', '1');
      check_equal(s, v, L'0', L'1');
    }
  }
  return true;
}

int main(int, char**)
{
  test_to_string<0>();
  test_to_string<1>();
  test_to_string<31>();
  test_to_string<32>();
  test_to_string<33>();
  test_to_string<63>();
  test_to_string<64>();
  test_to_string<65>();
  test_to_string<1000>(); // not in constexpr because of constexpr evaluation step limits
#  if TEST_STD_VER >= 2023
  static_assert(test_to_string<0>(), "");
  static_assert(test_to_string<1>(), "");
  static_assert(test_to_string<31>(), "");
  static_assert(test_to_string<32>(), "");
  static_assert(test_to_string<33>(), "");
  static_assert(test_to_string<63>(), "");
  static_assert(test_to_string<64>(), "");
  static_assert(test_to_string<65>(), "");
#  endif

  test_to_string_wchar<0>();
  test_to_string_wchar<1>();
  test_to_string_wchar<31>();
  test_to_string_wchar<32>();
  test_to_string_wchar<33>();
  test_to_string_wchar<63>();
  test_to_string_wchar<64>();
  test_to_string_wchar<65>();
  test_to_string_wchar<1000>(); // not in constexpr because of constexpr evaluation step limits
#  if TEST_STD_VER >= 2023
  static_assert(test_to_string_wchar<0>(), "");
  static_assert(test_to_string_wchar<1>(), "");
  static_assert(test_to_string_wchar<31>(), "");
  static_assert(test_to_string_wchar<32>(), "");
  static_assert(test_to_string_wchar<33>(), "");
  static_assert(test_to_string_wchar<63>(), "");
#  endif
  return 0;
}

#endif
