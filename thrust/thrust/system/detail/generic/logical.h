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
#include <thrust/logical.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
all_of(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, ::cuda::std::not_fn(pred)) == last;
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
any_of(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, pred) != last;
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE bool
none_of(thrust::execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return !thrust::any_of(exec, first, last, pred);
}

} // namespace generic
} // namespace detail
} // namespace system
THRUST_NAMESPACE_END
