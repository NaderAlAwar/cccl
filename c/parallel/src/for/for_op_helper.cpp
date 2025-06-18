//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>
// #include <format>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <util/types.h>

static std::string get_for_kernel_iterator(cccl_iterator_t iter)
{
  const auto input_it_value_t = cccl_type_enum_to_name(iter.value_type.type);
  const auto offset_t         = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

  constexpr std::string_view stateful_iterator =
    R"XXX(
extern "C" __device__ {3} {4}(const void *self_ptr);
extern "C" __device__ void {5}(void *self_ptr, {0} offset);
struct __align__({1}) input_iterator_state_t {{;
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type = {3};
  using difference_type = {0};
  using pointer = {3}*;
  using reference = {3}&;
  __device__ inline value_type operator*() const {{ return {4}(this); }}
  __device__ inline input_iterator_state_t& operator+=(difference_type diff) {{
      {5}(this, diff);
      return *this;
  }}
  __device__ inline value_type operator[](difference_type diff) const {{
      return *(*this + diff);
  }}
  __device__ inline input_iterator_state_t operator+(difference_type diff) const {{
      input_iterator_state_t result = *this;
      result += diff;
      return result;
  }}
  char data[{2}];
}};

using for_each_iterator_t = input_iterator_state_t;
)XXX";

  constexpr std::string_view stateless_iterator =
    R"XXX(
  using for_each_iterator_t = {0}*;
)XXX";

  if (iter.type == cccl_iterator_kind_t::CCCL_ITERATOR)
  {
    std::string result = stateful_iterator.data();
    result.replace(result.find("{0}"), 3, offset_t);
    result.replace(result.find("{1}"), 3, std::to_string(iter.alignment));
    result.replace(result.find("{2}"), 3, std::to_string(iter.size));
    result.replace(result.find("{3}"), 3, input_it_value_t);
    result.replace(result.find("{4}"), 3, iter.dereference.name);
    result.replace(result.find("{5}"), 3, iter.advance.name);
    return result;
  }
  else
  {
    std::string result = stateless_iterator.data();
    result.replace(result.find("{0}"), 3, input_it_value_t);
    return result;
  }
}

static std::string get_for_kernel_user_op(cccl_op_t user_op, cccl_iterator_t iter)
{
  auto value_t = cccl_type_enum_to_name(iter.value_type.type);

  constexpr std::string_view op_format =
    R"XXX(
#if {0}
#  define _STATEFUL_USER_OP
#endif

#define _USER_OP {1}
#define _USER_OP_INPUT_T {2}

#if defined(_STATEFUL_USER_OP)
extern "C" __device__ void _USER_OP(void*, _USER_OP_INPUT_T*);
#else
extern "C" __device__ void _USER_OP(_USER_OP_INPUT_T*);
#endif

#if defined(_STATEFUL_USER_OP)
struct __align__({3}) user_op_t {{
  char data[{4}];
#else
struct user_op_t {{
#endif

  __device__ void operator()(_USER_OP_INPUT_T* input) {{
#if defined(_STATEFUL_USER_OP)
    _USER_OP(&data, input);
#else
    _USER_OP(input);
#endif
  }}
}};
)XXX";

  bool user_op_stateful = cccl_op_kind_t::CCCL_STATEFUL == user_op.type;

  std::string result = op_format.data();
  result.replace(result.find("{0}"), 3, user_op_stateful ? "1" : "0");
  result.replace(result.find("{1}"), 3, user_op.name);
  result.replace(result.find("{2}"), 3, value_t);
  result.replace(result.find("{3}"), 3, std::to_string(user_op.alignment));
  result.replace(result.find("{4}"), 3, std::to_string(user_op.size));
  return result;
}

std::string get_for_kernel(cccl_op_t user_op, cccl_iterator_t iter)
{
  auto storage_align = iter.value_type.alignment;
  auto storage_size  = iter.value_type.size;

  std::string iterator_definition = get_for_kernel_iterator(iter);
  std::string user_op_definition  = get_for_kernel_user_op(user_op, iter);

  std::string result;
  result += "#include <cuda/std/iterator>\n";
  result += "#include <cub/agent/agent_for.cuh>\n";
  result += "#include <cub/device/dispatch/kernels/for_each.cuh>\n\n";

  result += "struct __align__(" + std::to_string(storage_align) + ") storage_t {\n";
  result += "  char data[" + std::to_string(storage_size) + "];\n";
  result += "};\n\n";

  // Append iterator wrapper
  result += iterator_definition + "\n\n";

  // Append user operator wrapper
  result += user_op_definition + "\n\n";

  result += "struct for_each_wrapper\n";
  result += "{\n";
  result += "  for_each_iterator_t iterator;\n";
  result += "  user_op_t user_op;\n\n";
  result += "  __device__ void operator()(unsigned long long idx)\n";
  result += "  {\n";
  result += "    user_op(iterator + idx);\n";
  result += "  }\n";
  result += "};\n\n";

  result += "using policy_dim_t = cub::detail::for_each::policy_t<256, 2>;\n\n";

  result += "struct device_for_policy\n";
  result += "{\n";
  result += "  struct ActivePolicy\n";
  result += "  {\n";
  result += "    using for_policy_t = policy_dim_t;\n";
  result += "  };\n";
  result += "};\n";

  return result;
}

constexpr static std::tuple<size_t, size_t>
calculate_kernel_state_sizes(size_t iter_size, size_t user_size, size_t user_align)
{
  size_t min_size       = iter_size;
  size_t user_op_offset = 0;

  if (user_size)
  {
    // Add space to match alignment provided by user
    size_t alignment = (min_size & (user_align - 1));
    if (alignment)
    {
      min_size += user_align - alignment;
    }
    // Capture offset where user function state begins
    user_op_offset = min_size;
    min_size += user_size;
  }

  return {min_size, user_op_offset};
}

static_assert(calculate_kernel_state_sizes(4, 8, 8) == std::tuple<size_t, size_t>{16, 8});
static_assert(calculate_kernel_state_sizes(2, 8, 8) == std::tuple<size_t, size_t>{16, 8});
static_assert(calculate_kernel_state_sizes(16, 8, 8) == std::tuple<size_t, size_t>{24, 16});
static_assert(calculate_kernel_state_sizes(8, 8, 8) == std::tuple<size_t, size_t>{16, 8});
static_assert(calculate_kernel_state_sizes(8, 16, 8) == std::tuple<size_t, size_t>{24, 8});
static_assert(calculate_kernel_state_sizes(8, 16, 16) == std::tuple<size_t, size_t>{32, 16});

for_each_kernel_state make_for_kernel_state(cccl_op_t op, cccl_iterator_t iterator)
{
  // Iterator is either a pointer or a stateful object, allocate space according to its size or alignment
  size_t iter_size     = (cccl_iterator_kind_t::CCCL_ITERATOR == iterator.type) ? iterator.size : sizeof(void*);
  void* iterator_state = (cccl_iterator_kind_t::CCCL_ITERATOR == iterator.type) ? iterator.state : &iterator.state;

  // Do we need to valid user input? Alignments larger than the provided size?
  size_t user_size  = (cccl_op_kind_t::CCCL_STATEFUL == op.type) ? op.size : 0;
  size_t user_align = (cccl_op_kind_t::CCCL_STATEFUL == op.type) ? op.alignment : 0;

  auto [min_size, user_op_offset] = calculate_kernel_state_sizes(iter_size, user_size, user_align);

  for_each_default local_buffer{};
  char* iter_start = (char*) &local_buffer;

  // Check if local blueprint provides enough space
  bool use_allocated_storage = sizeof(for_each_default) < min_size;

  if (use_allocated_storage)
  {
    // Allocate required space
    iter_start = new char[min_size];
  }

  // Memcpy into either local or allocated buffer
  memcpy(iter_start, iterator_state, iter_size);
  if (cccl_op_kind_t::CCCL_STATEFUL == op.type)
  {
    char* user_start = iter_start + user_op_offset;
    memcpy(user_start, op.state, user_size);
  }

  // Return either local buffer or unique_ptr
  if (use_allocated_storage)
  {
    return for_each_kernel_state{std::unique_ptr<char[]>{iter_start}, user_op_offset};
  }
  else
  {
    return for_each_kernel_state{local_buffer, user_op_offset};
  }
}

void* for_each_kernel_state::get()
{
  return std::visit(
    [](auto&& v) -> void* {
      using state_t = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<for_each_default, state_t>)
      {
        // Return the locally stored object as a void*
        return &v;
      }
      else
      {
        // Return the allocated space as a void*
        return v.get();
      }
    },
    for_each_arg);
}
