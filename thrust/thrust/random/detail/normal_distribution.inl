/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_real_distribution.h>

#include <cuda/std/cstdint>
#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN

namespace random
{

template <typename RealType>
_CCCL_HOST_DEVICE normal_distribution<RealType>::normal_distribution(RealType a, RealType b)
    : super_t()
    , m_param(a, b)
{} // end normal_distribution::normal_distribution()

template <typename RealType>
_CCCL_HOST_DEVICE normal_distribution<RealType>::normal_distribution(const param_type& parm)
    : super_t()
    , m_param(parm)
{} // end normal_distribution::normal_distribution()

template <typename RealType>
_CCCL_HOST_DEVICE void normal_distribution<RealType>::reset()
{
  super_t::reset();
} // end normal_distribution::reset()

template <typename RealType>
template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::result_type
normal_distribution<RealType>::operator()(UniformRandomNumberGenerator& urng)
{
  return operator()(urng, m_param);
} // end normal_distribution::operator()()

template <typename RealType>
template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::result_type
normal_distribution<RealType>::operator()(UniformRandomNumberGenerator& urng, const param_type& parm)
{
  return super_t::sample(urng, parm.first, parm.second);
} // end normal_distribution::operator()()

template <typename RealType>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::param_type normal_distribution<RealType>::param() const
{
  return m_param;
} // end normal_distribution::param()

template <typename RealType>
_CCCL_HOST_DEVICE void normal_distribution<RealType>::param(const param_type& parm)
{
  m_param = parm;
} // end normal_distribution::param()

template <typename RealType>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::result_type normal_distribution<RealType>::min
THRUST_PREVENT_MACRO_SUBSTITUTION() const
{
  return ::cuda::std::numeric_limits<RealType>::lowest();
} // end normal_distribution::min()

template <typename RealType>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::result_type normal_distribution<RealType>::max
THRUST_PREVENT_MACRO_SUBSTITUTION() const
{
  return ::cuda::std::numeric_limits<RealType>::max();
} // end normal_distribution::max()

template <typename RealType>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::result_type normal_distribution<RealType>::mean() const
{
  return m_param.first;
} // end normal_distribution::mean()

template <typename RealType>
_CCCL_HOST_DEVICE typename normal_distribution<RealType>::result_type normal_distribution<RealType>::stddev() const
{
  return m_param.second;
} // end normal_distribution::stddev()

template <typename RealType>
_CCCL_HOST_DEVICE bool normal_distribution<RealType>::equal(const normal_distribution& rhs) const
{
  return m_param == rhs.param();
}

template <typename RealType>
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& normal_distribution<RealType>::stream_out(std::basic_ostream<CharT, Traits>& os) const
{
  using ostream_type = std::basic_ostream<CharT, Traits>;
  using ios_base     = typename ostream_type::ios_base;

  // save old flags and fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill                        = os.fill();

  const CharT space = os.widen(' ');
  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(space);

  os << mean() << space << stddev();

  // restore old flags and fill character
  os.flags(flags);
  os.fill(fill);
  return os;
}

template <typename RealType>
template <typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& normal_distribution<RealType>::stream_in(std::basic_istream<CharT, Traits>& is)
{
  using istream_type = std::basic_istream<CharT, Traits>;
  using ios_base     = typename istream_type::ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::skipws);

  is >> m_param.first >> m_param.second;

  // restore old flags
  is.flags(flags);
  return is;
}

template <typename RealType>
_CCCL_HOST_DEVICE bool operator==(const normal_distribution<RealType>& lhs, const normal_distribution<RealType>& rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs, rhs);
}

template <typename RealType>
_CCCL_HOST_DEVICE bool operator!=(const normal_distribution<RealType>& lhs, const normal_distribution<RealType>& rhs)
{
  return !(lhs == rhs);
}

template <typename RealType, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const normal_distribution<RealType>& d)
{
  return thrust::random::detail::random_core_access::stream_out(os, d);
}

template <typename RealType, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is, normal_distribution<RealType>& d)
{
  return thrust::random::detail::random_core_access::stream_in(is, d);
}

} // namespace random

THRUST_NAMESPACE_END
