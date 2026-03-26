//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_select_if_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  CUkernel compact_init_kernel;
  CUkernel select_if_kernel;
  void* runtime_policy;
} cccl_device_select_if_build_result_t;

CCCL_C_API CUresult cccl_device_select_if_build(
  cccl_device_select_if_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

CCCL_C_API CUresult cccl_device_select_if_build_ex(
  cccl_device_select_if_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_select_flagged_if_build(
  cccl_device_select_if_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_flags,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

CCCL_C_API CUresult cccl_device_select_flagged_if_build_ex(
  cccl_device_select_if_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_flags,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_select_if(
  cccl_device_select_if_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  uint64_t num_items,
  CUstream stream);

CCCL_C_API CUresult cccl_device_select_flagged_if(
  cccl_device_select_if_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_flags,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  uint64_t num_items,
  CUstream stream);

CCCL_C_API CUresult cccl_device_select_if_cleanup(cccl_device_select_if_build_result_t* bld_ptr);
CCCL_C_API CUresult cccl_device_select_flagged_if_cleanup(cccl_device_select_if_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END
