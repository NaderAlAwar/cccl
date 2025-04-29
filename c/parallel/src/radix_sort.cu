//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_radix_sort.cuh>

// #include <format>

#include <filesystem>
#include <fstream>

#include "cccl/c/types.h"
#include "cub/util_type.cuh"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/radix_sort.h>
#include <nvrtc/ltoir_list_appender.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
struct agent_radix_sort_downsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct agent_radix_sort_upsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct agent_radix_sort_onesweep_policy
{
  int block_threads;
  int items_per_thread;
  int rank_num_parts;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct agent_radix_sort_histogram_policy
{
  int block_threads;
  int items_per_thread;
  int num_parts;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }
};

struct agent_radix_sort_exclusive_sum_policy
{
  int block_threads;
  int radix_bits;
};

struct agent_scan_policy
{
  int block_threads;
  int items_per_thread;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct radix_sort_runtime_tuning_policy
{
  agent_radix_sort_histogram_policy histogram;
  agent_radix_sort_exclusive_sum_policy exclusive_sum;
  agent_radix_sort_onesweep_policy onesweep;
  agent_scan_policy scan;
  agent_radix_sort_downsweep_policy downsweep;
  agent_radix_sort_downsweep_policy alt_downsweep;
  agent_radix_sort_upsweep_policy upsweep;
  agent_radix_sort_upsweep_policy alt_upsweep;
  agent_radix_sort_downsweep_policy single_tile;
  bool is_onesweep;

  agent_radix_sort_histogram_policy Histogram() const
  {
    return histogram;
  }

  agent_radix_sort_exclusive_sum_policy ExclusiveSum() const
  {
    return exclusive_sum;
  }

  agent_radix_sort_onesweep_policy Onesweep() const
  {
    return onesweep;
  }

  agent_scan_policy Scan() const
  {
    return scan;
  }

  agent_radix_sort_downsweep_policy Downsweep() const
  {
    return downsweep;
  }

  agent_radix_sort_downsweep_policy AltDownsweep() const
  {
    return alt_downsweep;
  }

  agent_radix_sort_upsweep_policy Upsweep() const
  {
    return upsweep;
  }

  agent_radix_sort_upsweep_policy AltUpsweep() const
  {
    return alt_upsweep;
  }

  agent_radix_sort_downsweep_policy SingleTile() const
  {
    return single_tile;
  }

  bool IsOnesweep() const
  {
    return is_onesweep;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int RadixBits(PolicyT policy)
  {
    return policy.radix_bits;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int BlockThreads(PolicyT policy)
  {
    return policy.block_threads;
  }
};

std::pair<int, int>
reg_bound_scaling(int nominal_4_byte_block_threads, int nominal_4_byte_items_per_thread, int key_size)
{
  assert(key_size > 0);
  int items_per_thread = std::max(1, nominal_4_byte_items_per_thread * 4 / std::max(4, key_size));
  int block_threads =
    std::min(nominal_4_byte_block_threads,
             cuda::ceil_div(int{cub::detail::max_smem_per_block} / (key_size * items_per_thread), 32) * 32);

  return {items_per_thread, block_threads};
}

std::pair<int, int>
mem_bound_scaling(int nominal_4_byte_block_threads, int nominal_4_byte_items_per_thread, int key_size)
{
  assert(key_size > 0);
  int items_per_thread =
    std::max(1, std::min(nominal_4_byte_items_per_thread * 4 / key_size, nominal_4_byte_items_per_thread * 2));
  int block_threads =
    std::min(nominal_4_byte_block_threads,
             cuda::ceil_div(int{cub::detail::max_smem_per_block} / (key_size * items_per_thread), 32) * 32);

  return {items_per_thread, block_threads};
}

radix_sort_runtime_tuning_policy get_policy(int /*cc*/, int key_size)
{
  // TODO: we hardcode some of these values in order to make sure that the radix_sort tests do not fail due to the
  // memory op assertions. This will be fixed after https://github.com/NVIDIA/cccl/issues/3570 is resolved.
  constexpr int onesweep_radix_bits = 8;
  const int primary_radix_bits      = (key_size > 1) ? 7 : 5;
  const int single_tile_radix_bits  = (key_size > 1) ? 6 : 5;
  // const bool offset_64bit           = sizeof(OffsetT) == 8;

  const agent_radix_sort_histogram_policy histogram_policy{
    128, 16, std::max(1, 1 * 4 / std::max(key_size, 4)), onesweep_radix_bits};
  constexpr agent_radix_sort_exclusive_sum_policy exclusive_sum_policy{256, onesweep_radix_bits};

  const auto [onesweep_items_per_thread, onesweep_block_threads] = reg_bound_scaling(384, 18, key_size);
  // const auto [scan_items_per_thread, scan_block_threads]         = mem_bound_scaling(512, 23, key_size);
  const int scan_items_per_thread = 5;
  const int scan_block_threads    = 512;
  // const auto [downsweep_items_per_thread, downsweep_block_threads] = mem_bound_scaling(160, 39, key_size);
  const int downsweep_items_per_thread = 5;
  const int downsweep_block_threads    = 160;
  // const auto [alt_downsweep_items_per_thread, alt_downsweep_block_threads] = mem_bound_scaling(256, 16, key_size);
  const int alt_downsweep_items_per_thread                             = 5;
  const int alt_downsweep_block_threads                                = 256;
  const auto [single_tile_items_per_thread, single_tile_block_threads] = mem_bound_scaling(256, 19, key_size);

  const bool is_onesweep = key_size >= static_cast<int>(sizeof(uint32_t));

  return {histogram_policy,
          exclusive_sum_policy,
          {onesweep_block_threads, onesweep_items_per_thread, 1, onesweep_radix_bits},
          {scan_block_threads, scan_items_per_thread},
          {downsweep_block_threads, downsweep_items_per_thread, primary_radix_bits},
          {alt_downsweep_block_threads, alt_downsweep_items_per_thread, primary_radix_bits - 1},
          {downsweep_block_threads, downsweep_items_per_thread, primary_radix_bits},
          {alt_downsweep_block_threads, alt_downsweep_items_per_thread, primary_radix_bits - 1},
          {single_tile_block_threads, single_tile_items_per_thread, single_tile_radix_bits},
          is_onesweep};
};

std::string get_single_tile_kernel_name(
  std::string_view chained_policy_t,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::DeviceRadixSortSingleTileKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending";
  result += ", ";
  result += std::string(key_t) + ", ";
  result += std::string(value_t) + ", ";
  result += std::string(offset_t) + ", ";
  result += "op_wrapper>";
  return result;
}

std::string get_upsweep_kernel_name(
  std::string_view chained_policy_t,
  bool alt_digit_bits,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::DeviceRadixSortUpsweepKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += alt_digit_bits ? "true" : "false";
  result += ", ";
  result += (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending";
  result += ", ";
  result += std::string(key_t) + ", ";
  result += std::string(offset_t) + ", ";
  result += "op_wrapper>";
  return result;
}

std::string get_scan_bins_kernel_name(std::string_view chained_policy_t, std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::RadixSortScanBinsKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += std::string(offset_t) + ">";
  return result;
}

std::string get_downsweep_kernel_name(
  std::string_view chained_policy_t,
  bool alt_digit_bits,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::DeviceRadixSortDownsweepKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += alt_digit_bits ? "true" : "false";
  result += ", ";
  result += (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending";
  result += ", ";
  result += std::string(key_t) + ", ";
  result += std::string(value_t) + ", ";
  result += std::string(offset_t) + ", ";
  result += "op_wrapper>";
  return result;
}

std::string get_histogram_kernel_name(
  std::string_view chained_policy_t,
  [[maybe_unused]] cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::DeviceRadixSortHistogramKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += "cub::SortOrder::Ascending, ";
  result += std::string(key_t) + ", ";
  result += std::string(offset_t) + ", ";
  result += "op_wrapper>";
  return result;
}

std::string get_exclusive_sum_kernel_name(std::string_view chained_policy_t, std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += std::string(offset_t) + ">";
  return result;
}

std::string get_onesweep_kernel_name(
  std::string_view chained_policy_t,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  std::string result = "cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<";
  result += std::string(chained_policy_t) + ", ";
  result += (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending";
  result += ", ";
  result += std::string(key_t) + ", ";
  result += std::string(value_t) + ", ";
  result += std::string(offset_t) + ", ";
  result += "int, int, op_wrapper>";
  return result;
}

template <auto* GetPolicy>
struct dynamic_radix_sort_policy_t
{
  using MaxPolicy = dynamic_radix_sort_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<radix_sort_runtime_tuning_policy>(GetPolicy(device_ptx_version, key_size));
  }

  uint64_t key_size;
};

struct radix_sort_kernel_source
{
  cccl_device_radix_sort_build_result_t& build;

  CUkernel RadixSortSingleTileKernel() const
  {
    return build.single_tile_kernel;
  }

  CUkernel RadixSortUpsweepKernel() const
  {
    return build.upsweep_kernel;
  }

  CUkernel RadixSortAltUpsweepKernel() const
  {
    return build.alt_upsweep_kernel;
  }

  CUkernel DeviceRadixSortScanBinsKernel() const
  {
    return build.scan_bins_kernel;
  }

  CUkernel RadixSortDownsweepKernel() const
  {
    return build.downsweep_kernel;
  }

  CUkernel RadixSortAltDownsweepKernel() const
  {
    return build.alt_downsweep_kernel;
  }

  CUkernel RadixSortHistogramKernel() const
  {
    return build.histogram_kernel;
  }

  CUkernel RadixSortExclusiveSumKernel() const
  {
    return build.exclusive_sum_kernel;
  }

  CUkernel RadixSortOnesweepKernel() const
  {
    return build.onesweep_kernel;
  }

  std::size_t KeySize() const
  {
    return build.key_type.size;
  }

  std::size_t ValueSize() const
  {
    return build.value_type.size;
  }
};

} // namespace radix_sort

static std::string inspect_sass(const void* cubin, size_t cubin_size)
{
  namespace fs = std::filesystem;

  fs::path temp_dir = fs::temp_directory_path();

  fs::path temp_in_filename  = temp_dir / "temp_in_file.cubin";
  fs::path temp_out_filename = temp_dir / "temp_out_file.sass";

  std::ofstream temp_in_file(temp_in_filename, std::ios::binary);
  if (!temp_in_file)
  {
    throw std::runtime_error("Failed to create temporary file.");
  }

  temp_in_file.write(static_cast<const char*>(cubin), cubin_size);
  temp_in_file.close();

  std::string command = "nvdisasm -gi ";
  command += temp_in_filename;
  command += " > ";
  command += temp_out_filename;

  int exec_code = std::system(command.c_str());

  if (!fs::remove(temp_in_filename))
  {
    throw std::runtime_error("Failed to remove temporary file.");
  }

  if (exec_code != 0)
  {
    throw std::runtime_error("Failed to execute command.");
  }

  std::ifstream temp_out_file(temp_out_filename, std::ios::binary);
  if (!temp_out_file)
  {
    throw std::runtime_error("Failed to create temporary file.");
  }

  const std::string sass{std::istreambuf_iterator<char>(temp_out_file), std::istreambuf_iterator<char>()};
  if (!fs::remove(temp_out_filename))
  {
    throw std::runtime_error("Failed to remove temporary file.");
  }

  return sass;
}

CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;
  try
  {
    const char* name = "test";

    const int cc       = cc_major * 10 + cc_minor;
    const auto policy  = radix_sort::get_policy(cc, input_keys_it.value_type.size);
    const auto key_cpp = cccl_type_enum_to_name(input_keys_it.value_type.type);
    const auto value_cpp =
      input_values_it.type == cccl_iterator_kind_t::CCCL_POINTER && input_values_it.state == nullptr
        ? "cub::NullType"
        : cccl_type_enum_to_name(input_values_it.value_type.type);
    const std::string op_src =
      (decomposer.name == nullptr || (decomposer.name != nullptr && decomposer.name[0] == '\0'))
        ? "using op_wrapper = cub::detail::identity_decomposer_t;"
        : make_kernel_user_unary_operator(key_cpp, decomposer_return_type, decomposer);
    constexpr std::string_view chained_policy_t = "device_radix_sort_policy";

    const std::string src_template = R"XXX(
#include <cub/device/dispatch/kernels/radix_sort.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>

struct __align__({1}) storage_t {
  char data[{0}];
};
struct __align__({3}) values_storage_t {
  char data[{2}];
};
struct agent_histogram_policy_t {
  static constexpr int ITEMS_PER_THREAD = {4};
  static constexpr int BLOCK_THREADS = {5};
  static constexpr int RADIX_BITS = {6};
  static constexpr int NUM_PARTS = {7};
};
struct agent_exclusive_sum_policy_t {
  static constexpr int BLOCK_THREADS = {8};
  static constexpr int RADIX_BITS = {9};
};
struct agent_onesweep_policy_t {
  static constexpr int ITEMS_PER_THREAD = {10};
  static constexpr int BLOCK_THREADS = {11};
  static constexpr int RANK_NUM_PARTS = {12};
  static constexpr int RADIX_BITS = {13};
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM       = cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM       = cub::BLOCK_SCAN_RAKING_MEMOIZE;
  static constexpr cub::RadixSortStoreAlgorithm STORE_ALGORITHM = cub::RADIX_SORT_STORE_DIRECT;
};
struct agent_scan_policy_t {
  static constexpr int ITEMS_PER_THREAD = {14};
  static constexpr int BLOCK_THREADS = {15};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = cub::LOAD_DEFAULT;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = cub::BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM   = cub::BLOCK_SCAN_RAKING_MEMOIZE;
  struct detail
  {
    using delay_constructor_t = cub::detail::default_delay_constructor_t<{16}>;
  };
};
struct agent_downsweep_policy_t {
  static constexpr int ITEMS_PER_THREAD = {17};
  static constexpr int BLOCK_THREADS = {18};
  static constexpr int RADIX_BITS = {19};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_BASIC;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
};
struct agent_alt_downsweep_policy_t {
  static constexpr int ITEMS_PER_THREAD = {20};
  static constexpr int BLOCK_THREADS = {21};
  static constexpr int RADIX_BITS = {22};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_RAKING_MEMOIZE;
};
struct agent_single_tile_policy_t {
  static constexpr int ITEMS_PER_THREAD = {23};
  static constexpr int BLOCK_THREADS = {24};
  static constexpr int RADIX_BITS = {25};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
};
struct {26} {
  struct ActivePolicy {
    using HistogramPolicy = agent_histogram_policy_t;
    using ExclusiveSumPolicy = agent_exclusive_sum_policy_t;
    using OnesweepPolicy = agent_onesweep_policy_t;
    using ScanPolicy = agent_scan_policy_t;
    using DownsweepPolicy = agent_downsweep_policy_t;
    using AltDownsweepPolicy = agent_alt_downsweep_policy_t;
    using UpsweepPolicy = agent_downsweep_policy_t;
    using AltUpsweepPolicy = agent_alt_downsweep_policy_t;
    using SingleTilePolicy = agent_single_tile_policy_t;
  };
};
{27};
)XXX";

    std::string offset_t;
    check(nvrtcGetTypeName<OffsetT>(&offset_t));

    std::string src = src_template;
    src.replace(src.find("{0}"), 3, std::to_string(input_keys_it.value_type.size));
    src.replace(src.find("{1}"), 3, std::to_string(input_keys_it.value_type.alignment));
    src.replace(src.find("{2}"), 3, std::to_string(input_values_it.value_type.size));
    src.replace(src.find("{3}"), 3, std::to_string(input_values_it.value_type.alignment));
    src.replace(src.find("{4}"), 3, std::to_string(policy.histogram.items_per_thread));
    src.replace(src.find("{5}"), 3, std::to_string(policy.histogram.block_threads));
    src.replace(src.find("{6}"), 3, std::to_string(policy.histogram.radix_bits));
    src.replace(src.find("{7}"), 3, std::to_string(policy.histogram.num_parts));
    src.replace(src.find("{8}"), 3, std::to_string(policy.exclusive_sum.block_threads));
    src.replace(src.find("{9}"), 3, std::to_string(policy.exclusive_sum.radix_bits));
    src.replace(src.find("{10}"), 4, std::to_string(policy.onesweep.items_per_thread));
    src.replace(src.find("{11}"), 4, std::to_string(policy.onesweep.block_threads));
    src.replace(src.find("{12}"), 4, std::to_string(policy.onesweep.rank_num_parts));
    src.replace(src.find("{13}"), 4, std::to_string(policy.onesweep.radix_bits));
    src.replace(src.find("{14}"), 4, std::to_string(policy.scan.items_per_thread));
    src.replace(src.find("{15}"), 4, std::to_string(policy.scan.block_threads));
    src.replace(src.find("{16}"), 4, offset_t);
    src.replace(src.find("{17}"), 4, std::to_string(policy.downsweep.items_per_thread));
    src.replace(src.find("{18}"), 4, std::to_string(policy.downsweep.block_threads));
    src.replace(src.find("{19}"), 4, std::to_string(policy.downsweep.radix_bits));
    src.replace(src.find("{20}"), 4, std::to_string(policy.alt_downsweep.items_per_thread));
    src.replace(src.find("{21}"), 4, std::to_string(policy.alt_downsweep.block_threads));
    src.replace(src.find("{22}"), 4, std::to_string(policy.alt_downsweep.radix_bits));
    src.replace(src.find("{23}"), 4, std::to_string(policy.single_tile.items_per_thread));
    src.replace(src.find("{24}"), 4, std::to_string(policy.single_tile.block_threads));
    src.replace(src.find("{25}"), 4, std::to_string(policy.single_tile.radix_bits));
    src.replace(src.find("{26}"), 4, std::string(chained_policy_t));
    src.replace(src.find("{27}"), 4, op_src);
#if true // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string single_tile_kernel_name =
      radix_sort::get_single_tile_kernel_name(chained_policy_t, sort_order, key_cpp, value_cpp, offset_t);
    std::string upsweep_kernel_name =
      radix_sort::get_upsweep_kernel_name(chained_policy_t, false, sort_order, key_cpp, offset_t);
    std::string alt_upsweep_kernel_name =
      radix_sort::get_upsweep_kernel_name(chained_policy_t, true, sort_order, key_cpp, offset_t);
    std::string scan_bins_kernel_name = radix_sort::get_scan_bins_kernel_name(chained_policy_t, offset_t);
    std::string downsweep_kernel_name =
      radix_sort::get_downsweep_kernel_name(chained_policy_t, false, sort_order, key_cpp, value_cpp, offset_t);
    std::string alt_downsweep_kernel_name =
      radix_sort::get_downsweep_kernel_name(chained_policy_t, true, sort_order, key_cpp, value_cpp, offset_t);
    std::string histogram_kernel_name =
      radix_sort::get_histogram_kernel_name(chained_policy_t, sort_order, key_cpp, offset_t);
    std::string exclusive_sum_kernel_name = radix_sort::get_exclusive_sum_kernel_name(chained_policy_t, offset_t);
    std::string onesweep_kernel_name =
      radix_sort::get_onesweep_kernel_name(chained_policy_t, sort_order, key_cpp, value_cpp, offset_t);
    std::string single_tile_kernel_lowered_name;
    std::string upsweep_kernel_lowered_name;
    std::string alt_upsweep_kernel_lowered_name;
    std::string scan_bins_kernel_lowered_name;
    std::string downsweep_kernel_lowered_name;
    std::string alt_downsweep_kernel_lowered_name;
    std::string histogram_kernel_lowered_name;
    std::string exclusive_sum_kernel_lowered_name;
    std::string onesweep_kernel_lowered_name;

    std::cout << "histogram_kernel_name: " << histogram_kernel_name << std::endl;

    const std::string arch = "-arch=sm_" + std::to_string(cc_major) + std::to_string(cc_minor);

    constexpr size_t num_args  = 8;
    const char* args[num_args] = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};
    appender.append({decomposer.ltoir, decomposer.ltoir_size});

    nvrtc_link_result result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({single_tile_kernel_name})
        .add_expression({upsweep_kernel_name})
        .add_expression({alt_upsweep_kernel_name})
        .add_expression({scan_bins_kernel_name})
        .add_expression({downsweep_kernel_name})
        .add_expression({alt_downsweep_kernel_name})
        .add_expression({histogram_kernel_name})
        .add_expression({exclusive_sum_kernel_name})
        .add_expression({onesweep_kernel_name})
        .compile_program({args, num_args})
        .get_name({single_tile_kernel_name, single_tile_kernel_lowered_name})
        .get_name({upsweep_kernel_name, upsweep_kernel_lowered_name})
        .get_name({alt_upsweep_kernel_name, alt_upsweep_kernel_lowered_name})
        .get_name({scan_bins_kernel_name, scan_bins_kernel_lowered_name})
        .get_name({downsweep_kernel_name, downsweep_kernel_lowered_name})
        .get_name({alt_downsweep_kernel_name, alt_downsweep_kernel_lowered_name})
        .get_name({histogram_kernel_name, histogram_kernel_lowered_name})
        .get_name({exclusive_sum_kernel_name, exclusive_sum_kernel_lowered_name})
        .get_name({onesweep_kernel_name, onesweep_kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    std::cout << "histogram_kernel_lowered_name: " << histogram_kernel_lowered_name << std::endl;

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(
      cuLibraryGetKernel(&build_ptr->single_tile_kernel, build_ptr->library, single_tile_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->upsweep_kernel, build_ptr->library, upsweep_kernel_lowered_name.c_str()));
    check(
      cuLibraryGetKernel(&build_ptr->alt_upsweep_kernel, build_ptr->library, alt_upsweep_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->scan_bins_kernel, build_ptr->library, scan_bins_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->downsweep_kernel, build_ptr->library, downsweep_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->alt_downsweep_kernel, build_ptr->library, alt_downsweep_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->histogram_kernel, build_ptr->library, histogram_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->exclusive_sum_kernel, build_ptr->library, exclusive_sum_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->onesweep_kernel, build_ptr->library, onesweep_kernel_lowered_name.c_str()));

    build_ptr->cc         = cc;
    build_ptr->cubin      = (void*) result.data.release();
    build_ptr->cubin_size = result.size;

    std::cout << inspect_sass(build_ptr->cubin, build_ptr->cubin_size) << std::endl;
    build_ptr->key_type   = input_keys_it.value_type;
    build_ptr->value_type = input_values_it.value_type;
    build_ptr->order      = sort_order;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

template <cub::SortOrder Order>
CUresult cccl_device_radix_sort_impl(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  printf("in second one\n");
  if (cccl_iterator_kind_t::CCCL_POINTER != d_keys_in.type || cccl_iterator_kind_t::CCCL_POINTER != d_values_in.type
      || cccl_iterator_kind_t::CCCL_POINTER != d_keys_out.type
      || cccl_iterator_kind_t::CCCL_POINTER != d_values_out.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_radix_sort(): radix sort input must be a pointer\n");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  CUresult error = CUDA_SUCCESS;
  printf("in third one\n");
  bool pushed = false;
  try
  {
    printf("in fourth one\n");
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));
    printf("in fifth one\n");

    indirect_arg_t key_arg_in{d_keys_in};
    indirect_arg_t key_arg_out{d_keys_out};
    cub::DoubleBuffer<indirect_arg_t> d_keys_buffer(
      *static_cast<indirect_arg_t**>(&key_arg_in), *static_cast<indirect_arg_t**>(&key_arg_out));

    printf("in sixth one\n");

    indirect_arg_t val_arg_in{d_values_in};
    indirect_arg_t val_arg_out{d_values_out};
    cub::DoubleBuffer<indirect_arg_t> d_values_buffer(
      *static_cast<indirect_arg_t**>(&val_arg_in), *static_cast<indirect_arg_t**>(&val_arg_out));

    printf("in seventh one\n");

    auto exec_status = cub::DispatchRadixSort<
      Order,
      indirect_arg_t,
      indirect_arg_t,
      OffsetT,
      indirect_arg_t,
      radix_sort::dynamic_radix_sort_policy_t<&radix_sort::get_policy>,
      radix_sort::radix_sort_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_keys_buffer,
        d_values_buffer,
        num_items,
        begin_bit,
        end_bit,
        is_overwrite_okay,
        stream,
        decomposer,
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        {d_keys_in.value_type.size});

    printf("in eighth one\n");

    *selector = d_keys_buffer.selector;
    error     = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  printf("in this one\n");
  return cccl_device_radix_sort_impl<cub::SortOrder::Ascending>(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    decomposer,
    num_items,
    begin_bit,
    end_bit,
    is_overwrite_okay,
    selector,
    stream);
}

CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
