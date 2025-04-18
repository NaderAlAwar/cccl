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

#include <thrust/detail/static_assert.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reduce.h>
#include <thrust/system/detail/generic/reduce.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename ExecutionPolicy, typename InputIterator>
_CCCL_HOST_DEVICE thrust::detail::it_value_t<InputIterator>
reduce(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last)
{
  using InputType = thrust::detail::it_value_t<InputIterator>;

  // use InputType(0) as init by default
  return thrust::reduce(exec, first, last, InputType(0));
} // end reduce()

template <typename ExecutionPolicy, typename InputIterator, typename T>
_CCCL_HOST_DEVICE T
reduce(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, T init)
{
  // use plus<T> by default
  return thrust::reduce(exec, first, last, init, ::cuda::std::plus<T>());
} // end reduce()

template <typename ExecutionPolicy, typename RandomAccessIterator, typename OutputType, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputType reduce(
  thrust::execution_policy<ExecutionPolicy>&, RandomAccessIterator, RandomAccessIterator, OutputType, BinaryFunction)
{
  static_assert(thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value,
                "unimplemented for this system");
  return OutputType();
} // end reduce()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
