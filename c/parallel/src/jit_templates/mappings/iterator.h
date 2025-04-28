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
#  include "util/types.h"
#endif

template <typename ValueTp>
struct cccl_iterator_t_mapping
{
  bool is_pointer                             = false;
  int size                                    = 1;
  int alignment                               = 1;
  void (*advance)(void*, cuda::std::uint64_t) = nullptr;
  ValueTp (*dereference)(const void*)         = nullptr;
  void (*assign)(const void*, ValueTp);

  using ValueT = ValueTp;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
struct output_iterator_traits;

template <>
struct parameter_mapping<cccl_iterator_t>
{
  static const constexpr auto archetype = cccl_iterator_t_mapping<int>{};

  template <typename Traits>
  static std::string map(template_id<Traits>, cccl_iterator_t arg)
  {
    std::string result = "cccl_iterator_t_mapping<";
    result += cccl_type_enum_to_name(arg.value_type.type);
    result += ">{.is_pointer = ";
    result += (arg.type == cccl_iterator_kind_t::CCCL_POINTER) ? "true" : "false";
    result += ", .size = ";
    result += std::to_string(arg.size);
    result += ", .alignment = ";
    result += std::to_string(arg.alignment);
    result += ", .advance = ";
    result += arg.advance.name;
    result += ", .";
    result += std::is_same_v<Traits, output_iterator_traits> ? "assign" : "dereference";
    result += " = ";
    result += arg.dereference.name;
    result += "}";
    return result;
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_iterator_t arg)
  {
    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      return std::string("extern \"C\" __device__ void ") + arg.advance.name + "(void *, "
           + cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64) + ");\n" + "extern \"C\" __device__ void "
           + arg.dereference.name + "(const void *, " + cccl_type_enum_to_name(arg.value_type.type) + ");\n";
    }

    return std::string("extern \"C\" __device__ void ") + arg.advance.name + "(void *, "
         + cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64) + ");\n" + "extern \"C\" __device__ "
         + cccl_type_enum_to_name(arg.value_type.type) + " " + arg.dereference.name + "(const void *);\n";
  }
};
#endif
