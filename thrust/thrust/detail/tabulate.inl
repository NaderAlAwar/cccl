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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/tabulate.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/tabulate.h>
#include <thrust/tabulate.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename UnaryOperation>
_CCCL_HOST_DEVICE void
tabulate(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         ForwardIterator first,
         ForwardIterator last,
         UnaryOperation unary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::tabulate");
  using thrust::system::detail::generic::tabulate;
  return tabulate(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, unary_op);
} // end tabulate()

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(ForwardIterator first, ForwardIterator last, UnaryOperation unary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::tabulate");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::tabulate(select_system(system), first, last, unary_op);
} // end tabulate()

THRUST_NAMESPACE_END
