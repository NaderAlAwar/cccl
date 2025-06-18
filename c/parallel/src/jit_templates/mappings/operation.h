//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include "../traits.h"
#  include "cccl/c/types.h"
#endif

struct cccl_op_t_mapping
{
  bool is_stateless   = false;
  int size            = 1;
  int alignment       = 1;
  void (*operation)() = nullptr;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
template <>
struct parameter_mapping<cccl_op_t>
{
  static const constexpr auto archetype = cccl_op_t_mapping{};

  template <typename Traits>
  static std::string map(template_id<Traits>, cccl_op_t op)
  {
    char buffer[256];
    std::snprintf(
      buffer,
      sizeof(buffer),
      "cccl_op_t_mapping{.is_stateless = %d, .size = %d, .alignment = %d, .operation = %s}",
      op.type == cccl_op_kind_t::CCCL_STATELESS,
      static_cast<int>(op.size),
      static_cast<int>(op.alignment),
      op.name);
    return std::string(buffer);
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_op_t op)
  {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "extern \"C\" __device__ void %s();\n", op.name);
    return std::string(buffer);
  }
};
#endif
