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
    return std::string("cccl_iterator_t_mapping<") + std::string(cccl_type_enum_to_name(arg.value_type.type))
         + std::string(">{.is_pointer = ") + std::to_string(arg.type == cccl_iterator_kind_t::CCCL_POINTER)
         + std::string(", .size = ") + std::to_string(arg.size) + std::string(", .alignment = ")
         + std::to_string(arg.alignment) + std::string(", .advance = ") + std::string(arg.advance.name)
         + std::string(", .")
         + (std::is_same_v<Traits, output_iterator_traits> ? std::string("assign") : std::string("dereference"))
         + std::string(" = ") + std::string(arg.dereference.name) + std::string("}");
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_iterator_t arg)
  {
    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      return std::string(R"output(
extern "C" __device__ void )output")
           + std::string(arg.advance.name) + std::string("(void *, ")
           + std::string(cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64)) + std::string(");\n")
           + std::string("extern \"C\" __device__ void ") + std::string(arg.dereference.name)
           + std::string("(const void *, ") + std::string(cccl_type_enum_to_name(arg.value_type.type))
           + std::string(");\n");
    }

    return std::string(R"input(
extern "C" __device__ void )input")
         + std::string(arg.advance.name) + std::string("(void *, ")
         + std::string(cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64)) + std::string(");\n")
         + std::string("extern \"C\" __device__ ") + std::string(cccl_type_enum_to_name(arg.value_type.type))
         + std::string(" ") + std::string(arg.dereference.name) + std::string("(const void *);\n");
  }
};
#endif
