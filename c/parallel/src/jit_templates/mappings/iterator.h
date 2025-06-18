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
    char buffer[256];
    std::string value_type_name = cccl_type_enum_to_name(arg.value_type.type);
    const char* advance_name    = arg.advance.name;
    const char* deref_or_assign = std::is_same_v<Traits, output_iterator_traits> ? "assign" : "dereference";
    const char* deref_name      = arg.dereference.name;
    std::snprintf(
      buffer,
      sizeof(buffer),
      "cccl_iterator_t_mapping<%s>{.is_pointer = %d, .size = %d, .alignment = %d, .advance = %s, .%s = %s}",
      value_type_name.c_str(),
      arg.type == cccl_iterator_kind_t::CCCL_POINTER,
      static_cast<int>(arg.size),
      static_cast<int>(arg.alignment),
      advance_name,
      deref_or_assign,
      deref_name);
    return std::string(buffer);
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_iterator_t arg)
  {
    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      char buffer[512];
      std::snprintf(
        buffer,
        sizeof(buffer),
        "extern \"C\" __device__ void %s(void *, %s);\nextern \"C\" __device__ void %s(const void *, %s);\n",
        arg.advance.name,
        cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64).c_str(),
        arg.dereference.name,
        cccl_type_enum_to_name(arg.value_type.type).c_str());
      return std::string(buffer);
    }

    char buffer[512];
    std::snprintf(
      buffer,
      sizeof(buffer),
      "extern \"C\" __device__ void %s(void *, %s);\nextern \"C\" __device__ %s %s(const void *);\n",
      arg.advance.name,
      cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64).c_str(),
      cccl_type_enum_to_name(arg.value_type.type).c_str(),
      arg.dereference.name);
    return std::string(buffer);
  }
};
#endif
