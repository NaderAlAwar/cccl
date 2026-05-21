// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>

#include "tensorrt_llm/kernels/customAllReduceKernels.h"

template <typename T>
constexpr nvinfer1::DataType trtllm_data_type()
{
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    return nvinfer1::DataType::kFLOAT;
  }
  else if constexpr (cuda::std::is_same_v<T, __half>)
  {
    return nvinfer1::DataType::kHALF;
  }
  else
  {
    static_assert(cuda::std::is_same_v<T, __nv_bfloat16>);
    return nvinfer1::DataType::kBF16;
  }
}

template <typename T>
inline void rms_norm_trtllm_reduce_fusion(
  T* d_input_and_residual_out,
  T const* d_residual,
  T* d_output,
  T const* d_weight,
  int num_tokens,
  int hidden_size,
  float eps,
  cudaStream_t stream = 0)
{
  tensorrt_llm::kernels::AllReduceParams params{};

  params.fusion_params.intermediate_buffer = d_input_and_residual_out;
  params.fusion_params.residual_buffer     = d_residual;
  params.fusion_params.weight_buffer       = d_weight;
  params.fusion_params.bias_buffer         = nullptr;
  params.fusion_params.hidden_size         = hidden_size;
  params.fusion_params.eps                 = eps;

  params.local_output_buffer_ptr = d_output;
  params.elts_total              = static_cast<std::size_t>(num_tokens) * static_cast<std::size_t>(hidden_size);

  tensorrt_llm::kernels::residualRmsNorm(
    params, trtllm_data_type<T>(), stream, tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM);
}
