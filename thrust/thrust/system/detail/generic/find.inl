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
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

#include <cuda/std/__algorithm/min.h>

// Contributed by Erich Elsen

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename InputIterator, typename T>
_CCCL_HOST_DEVICE InputIterator
find(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, const T& value)
{
  using thrust::placeholders::_1;

  return thrust::find_if(exec, first, last, _1 == value);
} // end find()

template <typename TupleType>
struct find_if_functor
{
  _CCCL_HOST_DEVICE TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if (thrust::get<0>(lhs) && thrust::get<0>(rhs))
    {
      return TupleType(true, (::cuda::std::min) (thrust::get<1>(lhs), thrust::get<1>(rhs)));
    }
    else if (thrust::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator
find_if(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  using difference_type = thrust::detail::it_difference_t<InputIterator>;
  using result_type     = typename thrust::tuple<bool, difference_type>;

  // empty sequence
  if (first == last)
  {
    return last;
  }

  const difference_type n = ::cuda::std::distance(first, last);

  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found

  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const difference_type interval_threshold = 1 << 20;
  const difference_type interval_size      = (::cuda::std::min) (interval_threshold, n);

  // force transform_iterator output to bool
  using XfrmIterator  = thrust::transform_iterator<Predicate, InputIterator, bool>;
  using IteratorTuple = thrust::tuple<XfrmIterator, thrust::counting_iterator<difference_type>>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  IteratorTuple iter_tuple =
    thrust::make_tuple(XfrmIterator(first, pred), thrust::counting_iterator<difference_type>(0));

  ZipIterator begin = thrust::make_zip_iterator(iter_tuple);
  ZipIterator end   = begin + n;

  for (ZipIterator interval_begin = begin; interval_begin < end; interval_begin += interval_size)
  {
    ZipIterator interval_end = interval_begin + interval_size;
    if (end < interval_end)
    {
      interval_end = end;
    } // end if

    result_type result = thrust::reduce(
      exec, interval_begin, interval_end, result_type(false, interval_end - begin), find_if_functor<result_type>());

    // see if we found something
    if (thrust::get<0>(result))
    {
      return first + thrust::get<1>(result);
    }
  }

  // nothing was found if we reach here...
  return first + n;
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator
find_if_not(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, ::cuda::std::not_fn(pred));
} // end find()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
