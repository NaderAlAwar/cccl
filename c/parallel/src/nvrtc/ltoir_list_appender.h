//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility> // std::move

#include "command_list.h"
#include <cccl/c/types.h>

struct nvrtc_linkable_list_appender
{
  nvrtc_linkable_list& linkable_list;

  void append(nvrtc_linkable linkable)
  {
    std::visit(
      [&](auto&& l) {
        if (l.size)
        {
          linkable_list.push_back(std::move(l));
        }
      },
      linkable);
  }

  // New method that handles both types
  void append_operation(cccl_op_t op)
  {
    // Append the primary code blob, honoring its type
    if (op.code_type == CCCL_OP_LTOIR)
    {
      append(nvrtc_linkable{nvrtc_ltoir{op.code, op.code_size}});
    }
    else
    {
      append(nvrtc_linkable{nvrtc_code{op.code, op.code_size}});
    }

    // Append any extra code blobs as LTO-IR units
    if (op.num_extra_code && op.extra_code && op.extra_code_sizes)
    {
      for (size_t i = 0; i < op.num_extra_code; ++i)
      {
        const char* blob = op.extra_code[i];
        size_t blob_size = op.extra_code_sizes[i];
        if (blob && blob_size)
        {
          printf("Appending extra code blob %p with size %zu\n", blob, blob_size);
          append(nvrtc_linkable{nvrtc_ltoir{blob, blob_size}});
        }
      }
    }
  }

  void add_iterator_definition(cccl_iterator_t it)
  {
    if (cccl_iterator_kind_t::CCCL_ITERATOR == it.type)
    {
      append_operation(it.advance); // Use new method
      append_operation(it.dereference); // Use new method
    }
  }
};
