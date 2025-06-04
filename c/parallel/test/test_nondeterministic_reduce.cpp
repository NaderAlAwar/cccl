//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <cstdint>

#include "test_util.h"

void reduce(cccl_iterator_t input, cccl_iterator_t output, uint64_t num_items, cccl_op_t op, cccl_value_t init)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_reduce_build_result_t build;
  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce_build(
            &build, input, output, op, init, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS == cccl_device_reduce(build, nullptr, &temp_storage_bytes, input, output, num_items, op, init, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(CUDA_SUCCESS
          == cccl_device_reduce(build, temp_storage.ptr, &temp_storage_bytes, input, output, num_items, op, init, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&build));
}

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
C2H_TEST("Reduce works with integral types", "[reduce]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(1);
  value_t<T> init{T{42}};

  reduce(input_ptr, output_ptr, num_items, op, init);

  const T output   = output_ptr[0];
  const T expected = std::accumulate(input.begin(), input.end(), init.value);
  REQUIRE(output == expected);
}
