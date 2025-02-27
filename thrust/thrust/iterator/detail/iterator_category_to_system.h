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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/iterator_categories.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
_CCCL_INLINE_VAR constexpr bool is_iterator_system =
  ::cuda::std::is_convertible_v<T, any_system_tag> || ::cuda::std::is_convertible_v<T, host_system_tag>
  || ::cuda::std::is_convertible_v<T, device_system_tag>;

// XXX this should work entirely differently
// we should just specialize this metafunction for iterator_category_with_system_and_traversal
template <typename Category>
struct iterator_category_to_system
{
  using type =
    // convertible to host iterator?
    ::cuda::std::_If<::cuda::std::is_convertible_v<Category, input_host_iterator_tag>
                       || ::cuda::std::is_convertible_v<Category, output_host_iterator_tag>,
                     host_system_tag,
                     // convertible to device iterator?
                     ::cuda::std::_If<::cuda::std::is_convertible_v<Category, input_device_iterator_tag>
                                        || ::cuda::std::is_convertible_v<Category, output_device_iterator_tag>,
                                      device_system_tag,
                                      // unknown system
                                      void>>;
};
} // namespace detail
THRUST_NAMESPACE_END
