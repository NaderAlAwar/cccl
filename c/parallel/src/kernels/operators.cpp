//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <string_view>

#include "cccl/c/types.h"
#include <kernels/operators.h>
#include <util/errors.h>
#include <util/types.h>

constexpr std::string_view binary_op_template = R"XXX(
#define LHS_T {0}
#define RHS_T {1}
#define OP_NAME {2}
#define OP_ALIGNMENT {3}
#define OP_SIZE {4}

// Source
{5}

#undef LHS_T
#undef RHS_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_SIZE
)XXX";

constexpr std::string_view stateless_binary_op_template = R"XXX(
extern "C" __device__ void OP_NAME(const void* lhs, const void* rhs, void* out);
struct op_wrapper {
  __device__ {0} operator()(LHS_T lhs, RHS_T rhs) const {
    {0} ret;
    OP_NAME(&lhs, &rhs, &ret);
    return ret;
  }
};
)XXX";

constexpr std::string_view stateful_binary_op_template = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {
  char data[OP_SIZE];
};
extern "C" __device__ void OP_NAME(void* state, const void* lhs, const void* rhs, void* out);
struct op_wrapper {
  op_state state;
  __device__ {0} operator()(LHS_T lhs, RHS_T rhs) {
    {0} ret;
    OP_NAME(&state, &lhs, &rhs, &ret);
    return ret;
  }
};
)XXX";

std::string make_kernel_binary_operator_full_source(
  std::string_view lhs_t, std::string_view rhs_t, cccl_op_t operation, std::string_view return_type)
{
  const std::string op_alignment =
    operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::to_string(operation.alignment);
  const std::string op_size = operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::to_string(operation.size);

  const std::string lhs_t_str(lhs_t);
  const std::string rhs_t_str(rhs_t);
  const std::string return_type_str(return_type);

  const std::string template_content =
    operation.type == cccl_op_kind_t::CCCL_STATELESS
      ? [&]() {
          std::string template_str(stateless_binary_op_template);
          size_t pos = 0;
          while ((pos = template_str.find("{0}", pos)) != std::string::npos)
          {
            template_str.replace(pos, 3, return_type_str);
            pos += return_type_str.length();
          }
          return template_str;
        }()
      : [&]() {
          std::string template_str(stateful_binary_op_template);
          size_t pos = 0;
          while ((pos = template_str.find("{0}", pos)) != std::string::npos)
          {
            template_str.replace(pos, 3, return_type_str);
            pos += return_type_str.length();
          }
          return template_str;
        }();

  std::string result = std::string(binary_op_template);
  result.replace(result.find("{5}"), 3, template_content);
  result.replace(result.find("{0}"), 3, lhs_t_str);
  result.replace(result.find("{1}"), 3, rhs_t_str);
  result.replace(result.find("{2}"), 3, std::string(operation.name));
  result.replace(result.find("{3}"), 3, op_alignment);
  result.replace(result.find("{4}"), 3, op_size);

  return result;
}

std::string make_kernel_user_binary_operator(
  std::string_view lhs_t, std::string_view rhs_t, std::string_view output_t, cccl_op_t operation)
{
  return make_kernel_binary_operator_full_source(lhs_t, rhs_t, operation, output_t);
}

std::string make_kernel_user_comparison_operator(std::string_view input_t, cccl_op_t operation)
{
  return make_kernel_binary_operator_full_source(input_t, input_t, operation, "bool");
}

std::string make_kernel_user_unary_operator(std::string_view input_t, std::string_view output_t, cccl_op_t operation)
{
  const std::string unary_op_template = R"XXX(
#define INPUT_T {0}
#define OUTPUT_T {1}
#define OP_NAME {2}
#define OP_ALIGNMENT {3}
#define OP_SIZE {4}

// Source
{5}

#undef INPUT_T
#undef OUTPUT_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_SIZE
)XXX";

  const std::string stateless_op = R"XXX(
extern "C" __device__  void OP_NAME(const void* val, void* result);
struct op_wrapper {
  __device__ OUTPUT_T operator()(INPUT_T val) const {
    OUTPUT_T out;
    OP_NAME(&val, &out);
    return out;
  }
};
)XXX";

  const std::string stateful_op = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {
  char data[OP_SIZE];
};
extern "C" __device__ void OP_NAME(op_state* state, const void* val, void* result);
struct op_wrapper
{
  op_state state;
  __device__ OUTPUT_T operator()(INPUT_T val)
  {
    OUTPUT_T out;
    OP_NAME(&state, &val, &out);
    return out;
  }
};

)XXX";

  const std::string input_t_str(input_t);
  const std::string output_t_str(output_t);
  const std::string op_name_str(operation.name);
  const std::string op_alignment =
    operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::to_string(operation.alignment);
  const std::string op_size = operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::to_string(operation.size);

  const std::string template_content = operation.type == cccl_op_kind_t::CCCL_STATELESS ? stateless_op : stateful_op;

  std::string result = unary_op_template;
  result.replace(result.find("{0}"), 3, input_t_str);
  result.replace(result.find("{1}"), 3, output_t_str);
  result.replace(result.find("{2}"), 3, op_name_str);
  result.replace(result.find("{3}"), 3, op_alignment);
  result.replace(result.find("{4}"), 3, op_size);
  result.replace(result.find("{5}"), 3, template_content);
  return result;
}
