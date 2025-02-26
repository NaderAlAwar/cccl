//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "test_util.h"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cccl/c/unique_by_key.h>

using key_types = std::tuple<uint8_t, int16_t, uint32_t, int64_t>;
using item_t    = int32_t;

void unique_by_key(
  cccl_iterator_t input_keys,
  cccl_iterator_t input_values,
  cccl_iterator_t output_keys,
  cccl_iterator_t output_values,
  cccl_iterator_t output_num_selected,
  cccl_op_t op,
  unsigned long long num_items)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_unique_by_key_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unique_by_key_build(
      &build,
      input_keys,
      input_values,
      output_keys,
      output_values,
      output_num_selected,
      op,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unique_by_key(
      build,
      nullptr,
      &temp_storage_bytes,
      input_keys,
      input_values,
      output_keys,
      output_values,
      output_num_selected,
      op,
      num_items,
      0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unique_by_key(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      input_keys,
      input_values,
      output_keys,
      output_values,
      output_num_selected,
      op,
      num_items,
      0));
  REQUIRE(CUDA_SUCCESS == cccl_device_unique_by_key_cleanup(&build));
}

TEMPLATE_LIST_TEST_CASE("DeviceSelect::UniqueByKey can run with empty input", "[unique_by_key]", key_types)
{
  constexpr int num_items = 0;

  operation_t op = make_operation("op", get_unique_by_key_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys(num_items);

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<int> output_num_selected_it(1);

  unique_by_key(input_keys_it, input_keys_it, input_keys_it, input_keys_it, output_num_selected_it, op, num_items);

  REQUIRE(0 == std::vector<int>(output_num_selected_it)[0]);
}

TEMPLATE_LIST_TEST_CASE("DeviceSelect::UniqueByKey works", "[unique_by_key]", key_types)
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op                   = make_operation("op", get_unique_by_key_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys = generate<TestType>(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<TestType> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  unique_by_key(input_keys_it, input_values_it, output_keys_it, output_values_it, output_num_selected_it, op, num_items);

  std::vector<std::pair<TestType, item_t>> input_pairs;
  for (size_t i = 0; i < input_keys.size(); ++i)
  {
    input_pairs.emplace_back(input_keys[i], input_values[i]);
  }
  const auto boundary = std::unique(input_pairs.begin(), input_pairs.end(), [](const auto& a, const auto& b) {
    return a.first == b.first;
  });

  int num_selected = std::vector<int>(output_num_selected_it)[0];

  REQUIRE((boundary - input_pairs.begin()) == num_selected);

  input_pairs.resize(num_selected);

  std::vector<TestType> host_output_keys(output_keys_it);
  std::vector<item_t> host_output_values(output_values_it);
  std::vector<std::pair<TestType, item_t>> output_pairs;
  for (int i = 0; i < num_selected; ++i)
  {
    output_pairs.emplace_back(host_output_keys[i], host_output_values[i]);
  }

  REQUIRE(input_pairs == output_pairs);
}

TEMPLATE_LIST_TEST_CASE("DeviceSelect::UniqueByKey handles none equal", "[device][select_unique_by_key]", key_types)
{
  const int num_items = 500; // to ensure that we get none equal for smaller data types

  operation_t op                   = make_operation("op", get_unique_by_key_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys = make_shuffled_sequence<TestType>(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<TestType> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  unique_by_key(input_keys_it, input_values_it, output_keys_it, output_values_it, output_num_selected_it, op, num_items);

  REQUIRE(num_items == std::vector<int>(output_num_selected_it)[0]);
  REQUIRE(input_keys == std::vector<TestType>(output_keys_it));
  REQUIRE(input_values == std::vector<item_t>(output_values_it));
}

TEMPLATE_LIST_TEST_CASE("DeviceSelect::UniqueByKey handles all equal", "[device][select_unique_by_key]", key_types)
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op = make_operation("op", get_unique_by_key_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys(num_items, static_cast<TestType>(1));
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<TestType> output_keys_it(1);
  pointer_t<item_t> output_values_it(1);
  pointer_t<int> output_num_selected_it(1);

  unique_by_key(input_keys_it, input_values_it, output_keys_it, output_values_it, output_num_selected_it, op, num_items);

  REQUIRE(1 == std::vector<int>(output_num_selected_it)[0]);
  REQUIRE(input_keys[0] == std::vector<TestType>(output_keys_it)[0]);
  REQUIRE(input_values[0] == std::vector<item_t>(output_values_it)[0]);
}
