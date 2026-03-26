//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/select_if.h>

using BuildResultT = cccl_device_select_if_build_result_t;

struct select_if_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_select_if_cleanup(build_data);
  }
};

using select_if_deleter       = BuildResultDeleter<BuildResultT, select_if_cleanup>;
using select_if_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, select_if_deleter>>;

struct select_flagged_if_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_select_flagged_if_cleanup(build_data);
  }
};

using select_flagged_if_deleter =
  BuildResultDeleter<BuildResultT, select_flagged_if_cleanup>;
using select_flagged_if_build_cache_t =
  build_cache_t<std::string, result_wrapper_t<BuildResultT, select_flagged_if_deleter>>;

template <typename CacheT, typename Tag>
auto& get_cache()
{
  return fixture<CacheT, Tag>::get_or_create().get_value();
}

struct select_if_build
{
  CUresult operator()(BuildResultT* build_ptr,
                      cccl_iterator_t d_in,
                      cccl_iterator_t d_selected_out,
                      cccl_iterator_t d_num_selected_out,
                      cccl_op_t select_op,
                      uint64_t,
                      int cc_major,
                      int cc_minor,
                      const char* cub_path,
                      const char* thrust_path,
                      const char* libcudacxx_path,
                      const char* ctk_path) const noexcept
  {
    return cccl_device_select_if_build(
      build_ptr,
      d_in,
      d_selected_out,
      d_num_selected_out,
      select_op,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path);
  }
};

struct select_if_run
{
  template <typename... Ts>
  CUresult operator()(Ts... args) const noexcept
  {
    return cccl_device_select_if(args...);
  }
};

struct select_flagged_if_build
{
  CUresult operator()(BuildResultT* build_ptr,
                      cccl_iterator_t d_in,
                      cccl_iterator_t d_flags,
                      cccl_iterator_t d_selected_out,
                      cccl_iterator_t d_num_selected_out,
                      cccl_op_t select_op,
                      uint64_t,
                      int cc_major,
                      int cc_minor,
                      const char* cub_path,
                      const char* thrust_path,
                      const char* libcudacxx_path,
                      const char* ctk_path) const noexcept
  {
    return cccl_device_select_flagged_if_build(
      build_ptr,
      d_in,
      d_flags,
      d_selected_out,
      d_num_selected_out,
      select_op,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path);
  }
};

struct select_flagged_if_run
{
  template <typename... Ts>
  CUresult operator()(Ts... args) const noexcept
  {
    return cccl_device_select_flagged_if(args...);
  }
};

template <typename BuildCache = select_if_build_cache_t, typename KeyT = std::string>
void select_if(cccl_iterator_t d_in,
               cccl_iterator_t d_selected_out,
               cccl_iterator_t d_num_selected_out,
               cccl_op_t select_op,
               uint64_t num_items,
               std::optional<BuildCache>& cache,
               const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, select_if_build, select_if_cleanup, select_if_run, BuildCache, KeyT>(
    cache, lookup_key, d_in, d_selected_out, d_num_selected_out, select_op, num_items);
}

template <typename BuildCache = select_flagged_if_build_cache_t, typename KeyT = std::string>
void select_flagged_if(cccl_iterator_t d_in,
                       cccl_iterator_t d_flags,
                       cccl_iterator_t d_selected_out,
                       cccl_iterator_t d_num_selected_out,
                       cccl_op_t select_op,
                       uint64_t num_items,
                       std::optional<BuildCache>& cache,
                       const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<
    BuildResultT,
    select_flagged_if_build,
    select_flagged_if_cleanup,
    select_flagged_if_run,
    BuildCache,
    KeyT>(cache, lookup_key, d_in, d_flags, d_selected_out, d_num_selected_out, select_op, num_items);
}

template <typename T, typename Predicate>
std::vector<T> std_select_if(const std::vector<T>& input, Predicate predicate)
{
  std::vector<T> output;
  output.reserve(input.size());
  std::copy_if(input.begin(), input.end(), std::back_inserter(output), predicate);
  return output;
}

template <typename T, typename FlagT, typename Predicate>
std::vector<T> std_select_flagged_if(const std::vector<T>& input,
                                     const std::vector<FlagT>& flags,
                                     Predicate predicate)
{
  std::vector<T> output;
  output.reserve(input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
  {
    if (predicate(flags[i]))
    {
      output.push_back(input[i]);
    }
  }
  return output;
}

template <typename T>
std::string get_select_if_op(cccl_type_enum type, int compare_to)
{
  return std::format(
    "#include <cuda_fp16.h>\n"
    "extern \"C\" __device__ void select_op(void* x_void, void* out_void) {{ "
    "  {0}* x = reinterpret_cast<{0}*>(x_void); "
    "  bool* out = reinterpret_cast<bool*>(out_void); "
    "  *out = *x < static_cast<{0}>({1}); "
    "}}",
    type_enum_to_name(type),
    compare_to);
}

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T& value) const
  {
    return value < compare;
  }
};

template <typename T, typename OperationT, typename TagT>
std::vector<T> c_parallel_select(OperationT select_op, const std::vector<T>& input)
{
  const std::size_t num_items = input.size();

  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(num_items);
  pointer_t<int> num_selected_ptr(1);

  auto& build_cache    = get_cache<select_if_build_cache_t, TagT>();
  const auto& test_key = make_key<T, int>();

  select_if(input_ptr, output_ptr, num_selected_ptr, select_op, num_items, build_cache, test_key);

  const int num_selected = num_selected_ptr[0];
  std::vector<T> output(output_ptr);
  output.resize(num_selected);
  return output;
}

template <typename T, typename FlagT, typename OperationT, typename TagT>
std::vector<T> c_parallel_select_flagged(OperationT select_op,
                                         const std::vector<T>& input,
                                         const std::vector<FlagT>& flags)
{
  const std::size_t num_items = input.size();

  pointer_t<T> input_ptr(input);
  pointer_t<FlagT> flags_ptr(flags);
  pointer_t<T> output_ptr(num_items);
  pointer_t<int> num_selected_ptr(1);

  auto& build_cache    = get_cache<select_flagged_if_build_cache_t, TagT>();
  const auto& test_key = make_key<T, FlagT, int>();

  select_flagged_if(input_ptr, flags_ptr, output_ptr, num_selected_ptr, select_op, num_items, build_cache, test_key);

  const int num_selected = num_selected_ptr[0];
  std::vector<T> output(output_ptr);
  output.resize(num_selected);
  return output;
}

using value_types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, std::int64_t, float, double>;
using flag_types  = c2h::type_list<std::uint8_t, std::int32_t, std::uint64_t>;

struct SelectIf_PrimitiveTypes_Fixture_Tag;
C2H_TEST("DeviceSelect::If works with primitive types", "[select_if]", value_types)
{
  using T = c2h::get<0, TestType>;

  constexpr int compare_to        = 21;
  operation_t select_op           = make_operation("select_op", get_select_if_op<T>(get_type_info<T>().type, compare_to));
  const std::size_t num_items     = GENERATE(0, 42, take(4, random(1 << 12, 1 << 20)));
  const std::vector<int> input_i  = generate<int>(num_items);
  const std::vector<T> input(input_i.begin(), input_i.end());

  auto c_parallel_result = c_parallel_select<T, operation_t, SelectIf_PrimitiveTypes_Fixture_Tag>(select_op, input);
  auto std_result        = std_select_if(input, less_than_t<T>{static_cast<T>(compare_to)});

  REQUIRE(c_parallel_result == std_result);
}

struct selector_state_t
{
  int comparison_value;
};

struct SelectIf_StatefulOperations_Fixture_Tag;
C2H_TEST("DeviceSelect::If works with stateful predicates", "[select_if]")
{
  selector_state_t op_state = {21};
  stateful_operation_t<selector_state_t> select_op = make_operation(
    "select_op",
    R"(struct selector_state_t { int comparison_value; };
extern "C" __device__ void select_op(void* state_ptr, void* x_ptr, void* out_ptr) {
  selector_state_t* state = static_cast<selector_state_t*>(state_ptr);
  *static_cast<bool*>(out_ptr) = *static_cast<int*>(x_ptr) < state->comparison_value;
})",
    op_state);

  const std::size_t num_items    = GENERATE(0, 42, take(4, random(1 << 12, 1 << 20)));
  const std::vector<int> input   = generate<int>(num_items);
  auto c_parallel_result =
    c_parallel_select<int, stateful_operation_t<selector_state_t>, SelectIf_StatefulOperations_Fixture_Tag>(
      select_op, input);
  auto std_result = std_select_if(input, less_than_t<int>{21});

  REQUIRE(c_parallel_result == std_result);
}

struct SelectIf_InputIterators_Fixture_Tag;
C2H_TEST("DeviceSelect::If works with input iterators", "[select_if]")
{
  constexpr int compare_to      = 17;
  const std::size_t num_items   = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t select_op         = make_operation("select_op", get_select_if_op<int>(get_type_info<int>().type, compare_to));
  auto input_it                 = make_counting_iterator<int>("int");
  input_it.state.value          = 0;
  pointer_t<int> output_it(num_items);
  pointer_t<int> num_selected(1);

  auto& build_cache    = get_cache<select_if_build_cache_t, SelectIf_InputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int, int>();

  select_if(input_it, output_it, num_selected, select_op, num_items, build_cache, test_key);

  const int count = num_selected[0];
  std::vector<int> output(output_it);
  output.resize(count);

  std::vector<int> expected;
  expected.reserve(compare_to);
  for (int i = 0; i < static_cast<int>(num_items) && i < compare_to; ++i)
  {
    expected.push_back(i);
  }

  REQUIRE(output == expected);
}

struct SelectFlaggedIf_PrimitiveTypes_Fixture_Tag;
C2H_TEST("DeviceSelect::FlaggedIf works with primitive types", "[select_if]", value_types, flag_types)
{
  using T     = c2h::get<0, TestType>;
  using FlagT = c2h::get<1, TestType>;

  constexpr int compare_to        = 21;
  operation_t select_op           = make_operation("select_op", get_select_if_op<FlagT>(get_type_info<FlagT>().type, compare_to));
  const std::size_t num_items     = GENERATE(0, 42, take(4, random(1 << 12, 1 << 20)));
  const std::vector<int> input_i  = generate<int>(num_items);
  const std::vector<int> flags_i  = generate<int>(num_items);
  const std::vector<T> input(input_i.begin(), input_i.end());
  const std::vector<FlagT> flags(flags_i.begin(), flags_i.end());

  auto c_parallel_result =
    c_parallel_select_flagged<T, FlagT, operation_t, SelectFlaggedIf_PrimitiveTypes_Fixture_Tag>(
      select_op, input, flags);
  auto std_result = std_select_flagged_if(input, flags, less_than_t<FlagT>{static_cast<FlagT>(compare_to)});

  REQUIRE(c_parallel_result == std_result);
}

struct SelectFlaggedIf_StatefulOperations_Fixture_Tag;
C2H_TEST("DeviceSelect::FlaggedIf works with stateful predicates", "[select_if]")
{
  selector_state_t op_state = {21};
  stateful_operation_t<selector_state_t> select_op = make_operation(
    "select_op",
    R"(struct selector_state_t { int comparison_value; };
extern "C" __device__ void select_op(void* state_ptr, void* flag_ptr, void* out_ptr) {
  selector_state_t* state = static_cast<selector_state_t*>(state_ptr);
  *static_cast<bool*>(out_ptr) = *static_cast<int*>(flag_ptr) < state->comparison_value;
})",
    op_state);

  const std::size_t num_items  = GENERATE(0, 42, take(4, random(1 << 12, 1 << 20)));
  const std::vector<int> input = generate<int>(num_items);
  const std::vector<int> flags = generate<int>(num_items);
  auto c_parallel_result =
    c_parallel_select_flagged<int, int, stateful_operation_t<selector_state_t>, SelectFlaggedIf_StatefulOperations_Fixture_Tag>(
      select_op, input, flags);
  auto std_result = std_select_flagged_if(input, flags, less_than_t<int>{21});

  REQUIRE(c_parallel_result == std_result);
}

struct SelectFlaggedIf_FlagsIterators_Fixture_Tag;
C2H_TEST("DeviceSelect::FlaggedIf works with flag iterators", "[select_if]")
{
  constexpr int compare_to      = 17;
  const std::size_t num_items   = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t select_op         = make_operation("select_op", get_select_if_op<int>(get_type_info<int>().type, compare_to));
  const std::vector<int> input  = generate<int>(num_items);
  pointer_t<int> input_it(input);
  auto flags_it                 = make_counting_iterator<int>("int");
  flags_it.state.value          = 0;
  pointer_t<int> output_it(num_items);
  pointer_t<int> num_selected(1);

  auto& build_cache    = get_cache<select_flagged_if_build_cache_t, SelectFlaggedIf_FlagsIterators_Fixture_Tag>();
  const auto& test_key = make_key<int, int, int>();

  select_flagged_if(input_it, flags_it, output_it, num_selected, select_op, num_items, build_cache, test_key);

  const int count = num_selected[0];
  std::vector<int> output(output_it);
  output.resize(count);

  std::vector<int> expected;
  expected.reserve(compare_to);
  for (int i = 0; i < static_cast<int>(num_items) && i < compare_to; ++i)
  {
    expected.push_back(input[static_cast<std::size_t>(i)]);
  }

  REQUIRE(output == expected);
}
