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
#include <thrust/detail/range/tail_flags.h>
#include <thrust/detail/seq.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/system/tbb/detail/execution_policy.h>
#include <thrust/system/tbb/detail/reduce_by_key.h>
#include <thrust/system/tbb/detail/reduce_intervals.h>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/cassert>

#include <thread>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{
namespace reduce_by_key_detail
{

template <typename L, typename R>
inline L divide_ri(const L x, const R y)
{
  return (x + (y - 1)) / y;
}

template <typename InputIterator, typename BinaryFunction, typename SFINAE = void>
struct partial_sum_type
{
  using type = thrust::detail::it_value_t<InputIterator>;
};

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate, typename BinaryFunction>
thrust::pair<InputIterator1,
             thrust::pair<thrust::detail::it_value_t<InputIterator1>,
                          typename partial_sum_type<InputIterator2, BinaryFunction>::type>>
reduce_last_segment_backward(
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  thrust::detail::it_difference_t<InputIterator1> n = keys_last - keys_first;

  // reverse the ranges and consume from the end
  thrust::reverse_iterator<InputIterator1> keys_first_r(keys_last);
  thrust::reverse_iterator<InputIterator1> keys_last_r(keys_first);
  thrust::reverse_iterator<InputIterator2> values_first_r(values_first + n);

  thrust::detail::it_value_t<InputIterator1> result_key                        = *keys_first_r;
  typename partial_sum_type<InputIterator2, BinaryFunction>::type result_value = *values_first_r;

  // consume the entirety of the first key's sequence
  for (++keys_first_r, ++values_first_r; (keys_first_r != keys_last_r) && binary_pred(*keys_first_r, result_key);
       ++keys_first_r, ++values_first_r)
  {
    result_value = binary_op(result_value, *values_first_r);
  }

  return thrust::make_pair(keys_first_r.base(), thrust::make_pair(result_key, result_value));
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
thrust::tuple<OutputIterator1,
              OutputIterator2,
              thrust::detail::it_value_t<InputIterator1>,
              typename partial_sum_type<InputIterator2, BinaryFunction>::type>
reduce_by_key_with_carry(
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  // first, consume the last sequence to produce the carry
  // XXX is there an elegant way to pose this such that we don't need to default construct carry?
  thrust::pair<thrust::detail::it_value_t<InputIterator1>,
               typename partial_sum_type<InputIterator2, BinaryFunction>::type>
    carry;

  thrust::tie(keys_last, carry) =
    reduce_last_segment_backward(keys_first, keys_last, values_first, binary_pred, binary_op);

  // finish with sequential reduce_by_key
  thrust::tie(keys_output, values_output) = thrust::reduce_by_key(
    thrust::seq, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);

  return thrust::make_tuple(keys_output, values_output, carry.first, carry.second);
}

template <typename Iterator>
bool interval_has_carry(size_t interval_idx, size_t interval_size, size_t num_intervals, Iterator tail_flags)
{
  // to discover whether the interval has a carry, look at the tail_flag corresponding to its last element
  // the final interval never has a carry by definition
  return (interval_idx + 1 < num_intervals) ? !tail_flags[(interval_idx + 1) * interval_size - 1] : false;
}

template <typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Iterator5,
          typename Iterator6,
          typename BinaryPredicate,
          typename BinaryFunction>
struct serial_reduce_by_key_body
{
  using size_type = thrust::detail::it_difference_t<Iterator1>;

  Iterator1 keys_first;
  Iterator2 values_first;
  Iterator3 result_offset;
  Iterator4 keys_result;
  Iterator5 values_result;
  Iterator6 carry_result;

  size_type n;
  size_type interval_size;
  size_type num_intervals;

  BinaryPredicate binary_pred;
  BinaryFunction binary_op;

  serial_reduce_by_key_body(
    Iterator1 keys_first,
    Iterator2 values_first,
    Iterator3 result_offset,
    Iterator4 keys_result,
    Iterator5 values_result,
    Iterator6 carry_result,
    size_type n,
    size_type interval_size,
    size_type num_intervals,
    BinaryPredicate binary_pred,
    BinaryFunction binary_op)
      : keys_first(keys_first)
      , values_first(values_first)
      , result_offset(result_offset)
      , keys_result(keys_result)
      , values_result(values_result)
      , carry_result(carry_result)
      , n(n)
      , interval_size(interval_size)
      , num_intervals(num_intervals)
      , binary_pred(binary_pred)
      , binary_op(binary_op)
  {}

  void operator()(const ::tbb::blocked_range<size_type>& r) const
  {
    assert(r.size() == 1);

    const size_type interval_idx = r.begin();

    const size_type offset_to_first = interval_size * interval_idx;
    const size_type offset_to_last  = ::cuda::std::min(n, offset_to_first + interval_size);

    Iterator1 my_keys_first    = keys_first + offset_to_first;
    Iterator1 my_keys_last     = keys_first + offset_to_last;
    Iterator2 my_values_first  = values_first + offset_to_first;
    Iterator3 my_result_offset = result_offset + interval_idx;
    Iterator4 my_keys_result   = keys_result + *my_result_offset;
    Iterator5 my_values_result = values_result + *my_result_offset;
    Iterator6 my_carry_result  = carry_result + interval_idx;

    // consume the rest of the interval with reduce_by_key
    using key_type   = thrust::detail::it_value_t<Iterator1>;
    using value_type = typename partial_sum_type<Iterator2, BinaryFunction>::type;

    // XXX is there a way to pose this so that we don't require default construction of carry?
    thrust::pair<key_type, value_type> carry;

    thrust::tie(my_keys_result, my_values_result, carry.first, carry.second) = reduce_by_key_with_carry(
      my_keys_first, my_keys_last, my_values_first, my_keys_result, my_values_result, binary_pred, binary_op);

    // store to carry only when we actually have a carry
    // store to my_keys_result & my_values_result otherwise

    // create tail_flags so we can check for a carry
    thrust::detail::tail_flags<Iterator1, BinaryPredicate> flags =
      thrust::detail::make_tail_flags(keys_first, keys_first + n, binary_pred);

    if (interval_has_carry(interval_idx, interval_size, num_intervals, flags.begin()))
    {
      // we can ignore the carry's key
      // XXX because the carry result is uninitialized, we should copy construct
      *my_carry_result = carry.second;
    }
    else
    {
      *my_keys_result   = carry.first;
      *my_values_result = carry.second;
    }
  }
};

template <typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Iterator5,
          typename Iterator6,
          typename BinaryPredicate,
          typename BinaryFunction>
serial_reduce_by_key_body<Iterator1, Iterator2, Iterator3, Iterator4, Iterator5, Iterator6, BinaryPredicate, BinaryFunction>
make_serial_reduce_by_key_body(
  Iterator1 keys_first,
  Iterator2 values_first,
  Iterator3 result_offset,
  Iterator4 keys_result,
  Iterator5 values_result,
  Iterator6 carry_result,
  thrust::detail::it_difference_t<Iterator1> n,
  size_t interval_size,
  size_t num_intervals,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  return serial_reduce_by_key_body<
    Iterator1,
    Iterator2,
    Iterator3,
    Iterator4,
    Iterator5,
    Iterator6,
    BinaryPredicate,
    BinaryFunction>(
    keys_first,
    values_first,
    result_offset,
    keys_result,
    values_result,
    carry_result,
    n,
    interval_size,
    num_intervals,
    binary_pred,
    binary_op);
}

} // namespace reduce_by_key_detail

template <typename DerivedPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename BinaryPredicate,
          typename BinaryFunction>
thrust::pair<Iterator3, Iterator4> reduce_by_key(
  thrust::tbb::execution_policy<DerivedPolicy>& exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  Iterator3 keys_result,
  Iterator4 values_result,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  using difference_type = thrust::detail::it_difference_t<Iterator1>;
  difference_type n     = keys_last - keys_first;
  if (n == 0)
  {
    return thrust::make_pair(keys_result, values_result);
  }

  // XXX this value is a tuning opportunity
  const difference_type parallelism_threshold = 10000;

  if (n < parallelism_threshold)
  {
    // don't bother parallelizing for small n
    return thrust::reduce_by_key(
      thrust::seq, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
  }

  // count the number of processors
  const unsigned int p = ::cuda::std::max<unsigned int>(1u, std::thread::hardware_concurrency());

  // generate O(P) intervals of sequential work
  // XXX oversubscribing is a tuning opportunity
  const unsigned int subscription_rate = 1;
  difference_type interval_size        = ::cuda::std::min<difference_type>(
    parallelism_threshold, ::cuda::std::max<difference_type>(n, n / (subscription_rate * p)));
  difference_type num_intervals = reduce_by_key_detail::divide_ri(n, interval_size);

  // decompose the input into intervals of size N / num_intervals
  // add one extra element to this vector to store the size of the entire result
  thrust::detail::temporary_array<difference_type, DerivedPolicy> interval_output_offsets(0, exec, num_intervals + 1);

  // first count the number of tail flags in each interval
  thrust::detail::tail_flags<Iterator1, BinaryPredicate> tail_flags =
    thrust::detail::make_tail_flags(keys_first, keys_last, binary_pred);
  thrust::system::tbb::detail::reduce_intervals(
    exec,
    tail_flags.begin(),
    tail_flags.end(),
    interval_size,
    interval_output_offsets.begin() + 1,
    ::cuda::std::plus<size_t>());
  interval_output_offsets[0] = 0;

  // scan the counts to get each body's output offset
  thrust::inclusive_scan(
    thrust::seq,
    interval_output_offsets.begin() + 1,
    interval_output_offsets.end(),
    interval_output_offsets.begin() + 1);

  // do a reduce_by_key serially in each thread
  // the final interval never has a carry by definition, so don't reserve space for it
  using carry_type = typename reduce_by_key_detail::partial_sum_type<Iterator2, BinaryFunction>::type;
  thrust::detail::temporary_array<carry_type, DerivedPolicy> carries(0, exec, num_intervals - 1);

  // force grainsize == 1 with simple_partioner()
  ::tbb::parallel_for(
    ::tbb::blocked_range<difference_type>(0, num_intervals, 1),
    reduce_by_key_detail::make_serial_reduce_by_key_body(
      keys_first,
      values_first,
      interval_output_offsets.begin(),
      keys_result,
      values_result,
      carries.begin(),
      n,
      interval_size,
      num_intervals,
      binary_pred,
      binary_op),
    ::tbb::simple_partitioner());

  difference_type size_of_result = interval_output_offsets[num_intervals];

  // sequentially accumulate the carries
  // note that the last interval does not have a carry
  // XXX find a way to express this loop via a sequential algorithm, perhaps reduce_by_key
  for (typename thrust::detail::temporary_array<carry_type, DerivedPolicy>::size_type i = 0; i < carries.size(); ++i)
  {
    // if our interval has a carry, then we need to sum the carry to the next interval's output offset
    // if it does not have a carry, then we need to ignore carry_value[i]
    if (reduce_by_key_detail::interval_has_carry(i, interval_size, num_intervals, tail_flags.begin()))
    {
      difference_type output_idx = interval_output_offsets[i + 1];

      values_result[output_idx] = binary_op(values_result[output_idx], carries[i]);
    }
  }

  return thrust::make_pair(keys_result + size_of_result, values_result + size_of_result);
}

} // namespace detail
} // namespace tbb
} // namespace system
THRUST_NAMESPACE_END
