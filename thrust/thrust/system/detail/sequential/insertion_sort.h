/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/function.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/copy_backward.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{

_CCCL_EXEC_CHECK_DISABLE
template <typename RandomAccessIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void insertion_sort(RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
{
  using value_type = thrust::detail::it_value_t<RandomAccessIterator>;

  if (first == last)
  {
    return;
  }

  // wrap comp
  thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};

  for (RandomAccessIterator i = first + 1; i != last; ++i)
  {
    value_type tmp = *i;

    if (wrapped_comp(tmp, *first))
    {
      // tmp is the smallest value encountered so far
      sequential::copy_backward(first, i, i + 1);

      *first = tmp;
    }
    else
    {
      // tmp is not the smallest value, can avoid checking for j == first
      RandomAccessIterator j = i;
      RandomAccessIterator k = i - 1;

      while (wrapped_comp(tmp, *k))
      {
        *j = *k;
        j  = k;
        --k;
      }

      *j = tmp;
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void insertion_sort_by_key(
  RandomAccessIterator1 first1, RandomAccessIterator1 last1, RandomAccessIterator2 first2, StrictWeakOrdering comp)
{
  using value_type1 = thrust::detail::it_value_t<RandomAccessIterator1>;
  using value_type2 = thrust::detail::it_value_t<RandomAccessIterator2>;

  if (first1 == last1)
  {
    return;
  }

  // wrap comp
  thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};

  RandomAccessIterator1 i1 = first1 + 1;
  RandomAccessIterator2 i2 = first2 + 1;

  for (; i1 != last1; ++i1, ++i2)
  {
    value_type1 tmp1 = *i1;
    value_type2 tmp2 = *i2;

    if (wrapped_comp(tmp1, *first1))
    {
      // tmp is the smallest value encountered so far
      sequential::copy_backward(first1, i1, i1 + 1);
      sequential::copy_backward(first2, i2, i2 + 1);

      *first1 = tmp1;
      *first2 = tmp2;
    }
    else
    {
      // tmp is not the smallest value, can avoid checking for j == first
      RandomAccessIterator1 j1 = i1;
      RandomAccessIterator1 k1 = i1 - 1;

      RandomAccessIterator2 j2 = i2;
      RandomAccessIterator2 k2 = i2 - 1;

      while (wrapped_comp(tmp1, *k1))
      {
        *j1 = *k1;
        *j2 = *k2;

        j1 = k1;
        j2 = k2;

        --k1;
        --k2;
      }

      *j1 = tmp1;
      *j2 = tmp2;
    }
  }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
