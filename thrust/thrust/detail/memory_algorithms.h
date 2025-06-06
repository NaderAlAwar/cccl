// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

// TODO: These need to be turned into proper Thrust algorithms (dispatch layer,
// backends, etc).

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/memory_wrapper.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/utility>

#include <nv/target>

#include <new>

THRUST_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////

template <typename T>
_CCCL_HOST_DEVICE void destroy_at(T* location) noexcept
{
  location->~T();
}

template <typename Allocator, typename T>
_CCCL_HOST_DEVICE void destroy_at(Allocator const& alloc, T* location) noexcept
{
  using traits =
    typename detail::allocator_traits<::cuda::std::remove_cvref_t<Allocator>>::template rebind_traits<T>::other;

  typename traits::allocator_type alloc_T(alloc);

  traits::destroy(alloc_T, location);
}

template <typename ForwardIt>
_CCCL_HOST_DEVICE ForwardIt destroy(ForwardIt first, ForwardIt last) noexcept
{
  for (; first != last; ++first)
  {
    destroy_at(::cuda::std::addressof(*first));
  }

  return first;
}

template <typename Allocator, typename ForwardIt>
_CCCL_HOST_DEVICE ForwardIt destroy(Allocator const& alloc, ForwardIt first, ForwardIt last) noexcept
{
  using T = detail::it_value_t<ForwardIt>;
  using traits =
    typename detail::allocator_traits<::cuda::std::remove_cvref_t<Allocator>>::template rebind_traits<T>::other;

  typename traits::allocator_type alloc_T(alloc);

  for (; first != last; ++first)
  {
    destroy_at(alloc_T, ::cuda::std::addressof(*first));
  }

  return first;
}

template <typename ForwardIt, typename Size>
_CCCL_HOST_DEVICE ForwardIt destroy_n(ForwardIt first, Size n) noexcept
{
  for (; n > 0; (void) ++first, --n)
  {
    destroy_at(::cuda::std::addressof(*first));
  }

  return first;
}

template <typename Allocator, typename ForwardIt, typename Size>
_CCCL_HOST_DEVICE ForwardIt destroy_n(Allocator const& alloc, ForwardIt first, Size n) noexcept
{
  using T = detail::it_value_t<ForwardIt>;
  using traits =
    typename detail::allocator_traits<::cuda::std::remove_cvref_t<Allocator>>::template rebind_traits<T>::other;

  typename traits::allocator_type alloc_T(alloc);

  for (; n > 0; (void) ++first, --n)
  {
    destroy_at(alloc_T, ::cuda::std::addressof(*first));
  }

  return first;
}

template <typename ForwardIt, typename... Args>
_CCCL_HOST_DEVICE void uninitialized_construct(ForwardIt first, ForwardIt last, Args const&... args)
{
  using T = detail::it_value_t<ForwardIt>;

  ForwardIt current = first;

  // No exceptions in CUDA.
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      try {
        for (; current != last; ++current)
        {
          ::new (static_cast<void*>(::cuda::std::addressof(*current))) T(args...);
        }
      } catch (...) {
        destroy(first, current);
        throw;
      }),
    (for (; current != last; ++current) { ::new (static_cast<void*>(::cuda::std::addressof(*current))) T(args...); }));
}

template <typename Allocator, typename ForwardIt, typename... Args>
void uninitialized_construct_with_allocator(Allocator const& alloc, ForwardIt first, ForwardIt last, Args const&... args)
{
  using T      = detail::it_value_t<ForwardIt>;
  using traits = typename detail::allocator_traits<::cuda::std::remove_cvref_t<Allocator>>::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  ForwardIt current = first;

  // No exceptions in CUDA.
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      try {
        for (; current != last; ++current)
        {
          traits::construct(alloc_T, ::cuda::std::addressof(*current), args...);
        }
      } catch (...) {
        destroy(alloc_T, first, current);
        throw;
      }),
    (for (; current != last; ++current) { traits::construct(alloc_T, ::cuda::std::addressof(*current), args...); }));
}

template <typename ForwardIt, typename Size, typename... Args>
void uninitialized_construct_n(ForwardIt first, Size n, Args const&... args)
{
  using T = detail::it_value_t<ForwardIt>;

  ForwardIt current = first;

  // No exceptions in CUDA.
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      try {
        for (; n > 0; ++current, --n)
        {
          ::new (static_cast<void*>(::cuda::std::addressof(*current))) T(args...);
        }
      } catch (...) {
        destroy(first, current);
        throw;
      }),
    (for (; n > 0; ++current, --n) { ::new (static_cast<void*>(::cuda::std::addressof(*current))) T(args...); }));
}

template <typename Allocator, typename ForwardIt, typename Size, typename... Args>
void uninitialized_construct_n_with_allocator(Allocator const& alloc, ForwardIt first, Size n, Args const&... args)
{
  using T      = detail::it_value_t<ForwardIt>;
  using traits = typename detail::allocator_traits<::cuda::std::remove_cvref_t<Allocator>>::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  ForwardIt current = first;

  // No exceptions in CUDA.
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      try {
        for (; n > 0; (void) ++current, --n)
        {
          traits::construct(alloc_T, ::cuda::std::addressof(*current), args...);
        }
      } catch (...) {
        destroy(alloc_T, first, current);
        throw;
      }),
    (for (; n > 0; (void) ++current, --n) { traits::construct(alloc_T, ::cuda::std::addressof(*current), args...); }));
}

///////////////////////////////////////////////////////////////////////////////

THRUST_NAMESPACE_END
