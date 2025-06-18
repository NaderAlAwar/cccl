//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// #include <format>

#include "cccl/c/types.h"
#include <kernels/iterators.h>
#include <util/errors.h>
#include <util/types.h>

constexpr std::string_view format_template = R"XXX(
#define DIFF_T {0}
#define OP_ALIGNMENT {1}
#define OP_SIZE {2}
#define VALUE_T {3}
#define DEREF {4}
#define ADVANCE {5}

// Kernel Source
{6}

#undef DIFF_T
#undef OP_ALIGNMENT
#undef OP_SIZE
#undef VALUE_T
#undef DEREF
#undef ADVANCE
)XXX";

std::string make_kernel_input_iterator(
  std::string_view diff_t,
  size_t alignment,
  size_t size,
  std::string_view iterator_name,
  std::string_view value_t,
  std::string_view deref,
  std::string_view advance)
{
  const std::string iter_def =
    R"XXX(
extern "C" __device__ VALUE_T DEREF(const void *self_ptr);
extern "C" __device__ void ADVANCE(void *self_ptr, DIFF_T offset);
struct __align__(OP_ALIGNMENT) )XXX"
    + std::string(iterator_name) + R"XXX( {
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type = VALUE_T;
  using difference_type = DIFF_T;
  using pointer = VALUE_T*;
  using reference = VALUE_T&;
  __device__ inline value_type operator*() const { return DEREF(data); }
  __device__ inline )XXX"
    + std::string(iterator_name) + R"XXX(& operator+=(difference_type diff) {
      ADVANCE(data, diff);
      return *this;
  }
  __device__ inline value_type operator[](difference_type diff) const {
      return *(*this + diff);
  }
  __device__ inline )XXX"
    + std::string(iterator_name) + R"XXX( operator+(difference_type diff) const {
      )XXX"
    + std::string(iterator_name) + R"XXX( result = *this;
      result += diff;
      return result;
  }
  char data[OP_SIZE];
};
)XXX";

  std::string result = std::string(format_template);
  result.replace(result.find("{0}"), 3, diff_t);
  result.replace(result.find("{1}"), 3, std::to_string(alignment));
  result.replace(result.find("{2}"), 3, std::to_string(size));
  result.replace(result.find("{3}"), 3, value_t);
  result.replace(result.find("{4}"), 3, deref);
  result.replace(result.find("{5}"), 3, advance);
  result.replace(result.find("{6}"), 3, iter_def);
  return result;
};

std::string make_kernel_input_iterator(
  std::string_view offset_t, std::string_view iterator_name, std::string_view input_value_t, cccl_iterator_t iter)
{
  if (iter.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return {};
  }

  return make_kernel_input_iterator(
    offset_t, iter.alignment, iter.size, iterator_name, input_value_t, iter.dereference.name, iter.advance.name);
}

std::string make_kernel_output_iterator(
  std::string_view diff_t,
  size_t alignment,
  size_t size,
  std::string_view iterator_name,
  std::string_view value_t,
  std::string_view deref,
  std::string_view advance)
{
  const std::string iter_def =
    R"XXX(
extern "C" __device__ void )XXX"
    + std::string(deref) + R"XXX((const void *self_ptr, VALUE_T x);
extern "C" __device__ void )XXX"
    + std::string(advance) + R"XXX((void *self_ptr, DIFF_T offset);
struct __align__(OP_ALIGNMENT) )XXX"
    + std::string(iterator_name) + R"XXX(_state_t {
  char data[OP_SIZE];
};
struct )XXX"
    + std::string(iterator_name) + R"XXX(_proxy_t {
  __device__ )XXX"
    + std::string(iterator_name) + R"XXX(_proxy_t operator=(VALUE_T x) {
    )XXX"
    + std::string(deref) + R"XXX((&state, x);
    return *this;
  }
  )XXX"
    + std::string(iterator_name) + R"XXX(_state_t state;
};
struct )XXX"
    + std::string(iterator_name) + R"XXX( {
  using iterator_category = cuda::std::random_access_iterator_tag;
  using difference_type   = DIFF_T;
  using value_type        = void;
  using pointer           = )XXX"
    + std::string(iterator_name) + R"XXX(_proxy_t*;
  using reference         = )XXX"
    + std::string(iterator_name) + R"XXX(_proxy_t;
  __device__ )XXX"
    + std::string(iterator_name) + R"XXX(_proxy_t operator*() const { return {state}; }
  __device__ )XXX"
    + std::string(iterator_name) + R"XXX(& operator+=(difference_type diff) {
      )XXX"
    + std::string(advance) + R"XXX((&state, diff);
      return *this;
  }
  __device__ )XXX"
    + std::string(iterator_name) + R"XXX(_proxy_t operator[](difference_type diff) const {
    )XXX"
    + std::string(iterator_name) + R"XXX( result = *this;
    result += diff;
    return { result.state };
  }
  __device__ )XXX"
    + std::string(iterator_name) + R"XXX( operator+(difference_type diff) const {
    )XXX"
    + std::string(iterator_name) + R"XXX( result = *this;
    result += diff;
    return result;
  }
  )XXX"
    + std::string(iterator_name) + R"XXX(_state_t state;
};
)XXX";

  std::string result = std::string(format_template);
  result.replace(result.find("{0}"), 3, diff_t);
  result.replace(result.find("{1}"), 3, std::to_string(alignment));
  result.replace(result.find("{2}"), 3, std::to_string(size));
  result.replace(result.find("{3}"), 3, value_t);
  result.replace(result.find("{4}"), 3, deref);
  result.replace(result.find("{5}"), 3, advance);
  result.replace(result.find("{6}"), 3, iter_def);
  return result;
};

std::string make_kernel_output_iterator(
  std::string_view offset_t, std::string_view iterator_name, std::string_view input_value_t, cccl_iterator_t iter)
{
  if (iter.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return {};
  }

  return make_kernel_output_iterator(
    offset_t, iter.alignment, iter.size, iterator_name, input_value_t, iter.dereference.name, iter.advance.name);
}

std::string make_kernel_inout_iterator(
  std::string_view diff_t,
  size_t alignment,
  size_t size,
  std::string_view value_t,
  std::string_view deref,
  std::string_view advance)
{
  const std::string iter_def =
    R"XXX(
extern "C" __device__ VALUE_T* DEREF(const void *self_ptr);
extern "C" __device__ void ADVANCE(void *self_ptr, DIFF_T offset);

struct __align__(OP_ALIGNMENT) output_iterator_state_t {
  char data[OP_SIZE];
};

struct output_iterator_t {
  using iterator_category = cuda::std::random_access_iterator_tag;
  using difference_type   = DIFF_T;
  using value_type        = VALUE_T;
  using pointer           = output_iterator_proxy_t*;
  using reference         = output_iterator_proxy_t;
  __device__ value_type* operator*() const { return DEREF(&state); }
  __device__ output_iterator_t& operator+=(difference_type diff) {
      ADVANCE(&state, diff);
      return *this;
  }
  __device__ output_iterator_proxy_t operator[](difference_type diff) const {
    output_iterator_t result = *this;
    result += diff;
    return { result.state };
  }
  __device__ output_iterator_t operator+(difference_type diff) const {
    output_iterator_t result = *this;
    result += diff;
    return result;
  }
  output_iterator_state_t state;
};
)XXX";

  std::string result = std::string(format_template);
  result.replace(result.find("{0}"), 3, diff_t);
  result.replace(result.find("{1}"), 3, value_t);
  result.replace(result.find("{2}"), 3, deref);
  result.replace(result.find("{3}"), 3, advance);
  result.replace(result.find("{4}"), 3, std::to_string(size));
  result.replace(result.find("{5}"), 3, std::to_string(alignment));
  result.replace(result.find("{6}"), 3, iter_def);

  result.replace(result.find("{0}"), 3, diff_t);
  result.replace(result.find("{1}"), 3, std::to_string(alignment));
  result.replace(result.find("{2}"), 3, std::to_string(size));
  result.replace(result.find("{3}"), 3, value_t);
  result.replace(result.find("{4}"), 3, deref);
  result.replace(result.find("{5}"), 3, advance);
  return result;
};

std::string make_kernel_inout_iterator(std::string_view offset_t, std::string_view input_value_t, cccl_iterator_t iter)
{
  if (iter.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return {};
  }

  return make_kernel_inout_iterator(
    offset_t, iter.alignment, iter.size, input_value_t, iter.dereference.name, iter.advance.name);
}
