//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/detail/ptx-json-parser.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>

#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/operation.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include <cccl/c/select_if.h>
#include <nlohmann/json.hpp>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/types.h>

struct device_select_if_policy;

using OffsetT = ptrdiff_t;
static_assert(std::is_same_v<cub::detail::choose_signed_offset_t<OffsetT>, OffsetT>,
              "OffsetT must be signed int32 or int64");

struct input_storage_t;
struct flags_storage_t;
struct selected_output_storage_t;
struct num_selected_output_storage_t;

struct select_if_input_iterator_tag;
struct select_if_flags_iterator_tag;
struct select_if_output_iterator_tag;
struct select_if_num_selected_output_iterator_tag;
struct select_if_operation_tag;
struct select_flagged_if_operation_tag;

namespace select_if
{
template <bool UsesFlags>
inline constexpr const char* api_name_v = UsesFlags ? "device_select_flagged_if" : "device_select_if";

template <bool UsesFlags>
using dispatch_flags_iterator_t = std::conditional_t<UsesFlags, indirect_arg_t, cub::NullType*>;

struct select_if_runtime_tuning_policy
{
  cub::detail::RuntimeSelectIfAgentPolicy select_if;
  ::cuda::std::size_t vsmem_per_block{};

  auto SelectIf() const
  {
    return *this;
  }

  int BlockThreads() const
  {
    return select_if.BlockThreads();
  }

  int ItemsPerThread() const
  {
    return select_if.ItemsPerThread();
  }

  ::cuda::std::size_t VSMemPerBlock() const
  {
    return vsmem_per_block;
  }

  using SelectIfPolicyT = select_if_runtime_tuning_policy;
  using MaxPolicy       = select_if_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<select_if_runtime_tuning_policy>(*this);
  }
};

struct select_if_kernel_source
{
  cccl_device_select_if_build_result_t& build;

  CUkernel CompactInitKernel() const
  {
    return build.compact_init_kernel;
  }

  CUkernel SelectIfKernel() const
  {
    return build.select_if_kernel;
  }
};

struct dynamic_vsmem_helper_t
{
  template <typename PolicyT, typename... Ts>
  static int BlockThreads(PolicyT policy)
  {
    return policy.BlockThreads();
  }

  template <typename PolicyT, typename... Ts>
  static int ItemsPerThread(PolicyT policy)
  {
    return policy.ItemsPerThread();
  }

  template <typename PolicyT, typename... Ts>
  static ::cuda::std::size_t VSMemPerBlock(PolicyT policy)
  {
    return policy.VSMemPerBlock();
  }
};

std::string make_storage_definition(std::string_view storage_name, cccl_type_info type_info)
{
  return std::format(
    "struct __align__({1}) {0} {{\n"
    "  char data[{2}];\n"
    "}};\n",
    storage_name,
    type_info.alignment,
    type_info.size);
}

std::string get_compact_init_kernel_name(std::string_view num_selected_out_iterator_name)
{
  std::string partition_offset_t;
  check(cccl_type_name_from_nvrtc<cub::detail::select::per_partition_offset_t>(&partition_offset_t));

  return std::format(
    "cub::detail::scan::DeviceCompactInitKernel<cub::ScanTileState<{0}>, {1}>",
    partition_offset_t,
    num_selected_out_iterator_name);
}

std::string get_select_if_kernel_name(std::string_view d_in_iterator_name,
                                      std::string_view d_flags_iterator_name,
                                      std::string_view d_selected_out_iterator_name,
                                      std::string_view d_num_selected_out_iterator_name,
                                      std::string_view select_op_name)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_select_if_policy>(&chained_policy_t));

  std::string global_offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&global_offset_t));

  std::string partition_offset_t;
  check(cccl_type_name_from_nvrtc<cub::detail::select::per_partition_offset_t>(&partition_offset_t));

  const auto scan_tile_state_t = std::format("cub::ScanTileState<{}>", partition_offset_t);
  const auto streaming_context_t =
    std::format("cub::detail::select::streaming_context_t<{0}, true>", global_offset_t);

  return std::format(
    "cub::detail::select::DeviceSelectSweepKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, cub::NullType, {7}, {8}, "
    "cub::SelectImpl::Select>",
    chained_policy_t,
    d_in_iterator_name,
    d_flags_iterator_name,
    d_selected_out_iterator_name,
    d_num_selected_out_iterator_name,
    scan_tile_state_t,
    select_op_name,
    partition_offset_t,
    streaming_context_t);
}
template <bool UsesFlags>
dispatch_flags_iterator_t<UsesFlags> make_dispatch_flags_iterator(cccl_iterator_t& d_flags)
{
  if constexpr (UsesFlags)
  {
    return indirect_arg_t{d_flags};
  }
  else
  {
    return static_cast<cub::NullType*>(nullptr);
  }
}

template <bool UsesFlags>
CUresult build_ex(cccl_device_select_if_build_result_t* build_ptr,
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
                  cccl_build_config* config)
try
{
  const char* const name           = api_name_v<UsesFlags>;
  const std::string policy_json_id = std::format("{}_policy", name);
  const int cc                     = cc_major * 10 + cc_minor;

  const auto [d_in_iterator_name, d_in_iterator_src] =
    get_specialization<select_if_input_iterator_tag, input_iterator_traits>(
      template_id<input_iterator_traits>(), tagged_arg<input_storage_t, cccl_iterator_t>{d_in});
  std::string d_flags_iterator_name{"cub::NullType*"};
  std::string d_flags_iterator_src;
  if constexpr (UsesFlags)
  {
    auto [flags_iterator_name, flags_iterator_src] =
      get_specialization<select_if_flags_iterator_tag, input_iterator_traits>(
        template_id<input_iterator_traits>(), tagged_arg<flags_storage_t, cccl_iterator_t>{d_flags});
    d_flags_iterator_name = std::move(flags_iterator_name);
    d_flags_iterator_src  = std::move(flags_iterator_src);
  }
  const auto [d_selected_out_iterator_name, d_selected_out_iterator_src] =
    get_specialization<select_if_output_iterator_tag, output_iterator_traits>(
      template_id<output_iterator_traits>(),
      tagged_arg<selected_output_storage_t, cccl_iterator_t>{d_selected_out},
      tagged_arg<selected_output_storage_t, cccl_type_info>{d_selected_out.value_type});
  const auto [d_num_selected_out_iterator_name, d_num_selected_out_iterator_src] =
    get_specialization<select_if_num_selected_output_iterator_tag, output_iterator_traits>(
      template_id<output_iterator_traits>(),
      tagged_arg<num_selected_output_storage_t, cccl_iterator_t>{d_num_selected_out},
      tagged_arg<num_selected_output_storage_t, cccl_type_info>{d_num_selected_out.value_type});

  cccl_type_info selector_result_t{sizeof(bool), alignof(bool), cccl_type_enum::CCCL_BOOLEAN};
  std::string select_op_name;
  std::string select_op_src;
  if constexpr (UsesFlags)
  {
    auto [name_v, src_v] = get_specialization<select_flagged_if_operation_tag, user_operation_traits>(
      template_id<user_operation_traits>(),
      select_op,
      selector_result_t,
      tagged_arg<flags_storage_t, cccl_type_info>{d_flags.value_type});
    select_op_name = std::move(name_v);
    select_op_src  = std::move(src_v);
  }
  else
  {
    auto [name_v, src_v] = get_specialization<select_if_operation_tag, user_operation_traits>(
      template_id<user_operation_traits>(),
      select_op,
      selector_result_t,
      tagged_arg<input_storage_t, cccl_type_info>{d_in.value_type});
    select_op_name = std::move(name_v);
    select_op_src  = std::move(src_v);
  }

  const auto input_t       = cccl_type_enum_to_name<input_storage_t>(d_in.value_type.type);
  const auto flags_value_t = [&] {
    if constexpr (UsesFlags)
    {
      return cccl_type_enum_to_name<flags_storage_t>(d_flags.value_type.type);
    }
    else
    {
      return std::string{"cub::NullType"};
    }
  }();
  const auto policy_hub_expr = std::format(
    "cub::detail::select::policy_hub<{0}, {1}, cub::detail::select::per_partition_offset_t, false, "
    "cub::SelectImpl::Select>",
    input_t,
    flags_value_t);

  std::string global_offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&global_offset_t));

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_select_if.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_select_if.cuh>
{0}
{1}
{2}
{3}
{4}
{5}
{6}
{7}
{8}
{9}
using device_select_if_policy = {10}::MaxPolicy;
using select_if_streaming_context_t = cub::detail::select::streaming_context_t<{11}, true>;
using select_if_vsmem_helper_t =
  cub::detail::select::VSMemHelper<cub::SelectImpl::Select>::template VSMemHelperDefaultFallbackPolicyT<
    device_select_if_policy::ActivePolicy::SelectIfPolicyT,
    {12},
    {13},
    {14},
    {15},
    cub::NullType,
    cub::detail::select::per_partition_offset_t,
    select_if_streaming_context_t>;

#include <cub/detail/ptx-json/json.cuh>
__device__ consteval auto& policy_generator() {{
  using namespace ptx_json;
  return ptx_json::id<ptx_json::string("{16}")>() = object<
    key<"Policy">() = cub::detail::select::SelectIfPolicyWrapper<device_select_if_policy::ActivePolicy>::EncodedPolicy(),
    key<"VSMemPerBlock">() = value<static_cast<int>(select_if_vsmem_helper_t::vsmem_per_block)>()>();
}}
)XXX",
    jit_template_header_contents,
    make_storage_definition("input_storage_t", d_in.value_type),
    UsesFlags ? make_storage_definition("flags_storage_t", d_flags.value_type) : std::string{},
    make_storage_definition("selected_output_storage_t", d_selected_out.value_type),
    make_storage_definition("num_selected_output_storage_t", d_num_selected_out.value_type),
    d_in_iterator_src,
    d_flags_iterator_src,
    d_selected_out_iterator_src,
    d_num_selected_out_iterator_src,
    select_op_src,
    policy_hub_expr,
    global_offset_t,
    d_in_iterator_name,
    d_flags_iterator_name,
    d_selected_out_iterator_name,
    select_op_name,
    policy_json_id);

  std::string compact_init_kernel_name =
    select_if::get_compact_init_kernel_name(d_num_selected_out_iterator_name);
  std::string select_if_kernel_name =
    select_if::get_select_if_kernel_name(d_in_iterator_name,
                                         d_flags_iterator_name,
                                         d_selected_out_iterator_name,
                                         d_num_selected_out_iterator_name,
                                         select_op_name);
  std::string compact_init_kernel_lowered_name;
  std::string select_if_kernel_lowered_name;

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  std::vector<const char*> args = {
    arch.c_str(),
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    "-rdc=true",
    "-dlto",
    "-DCUB_DISABLE_CDP",
    "-DCUB_ENABLE_POLICY_PTX_JSON",
    "-std=c++20"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(select_op);
  appender.add_iterator_definition(d_in);
  if constexpr (UsesFlags)
  {
    appender.add_iterator_definition(d_flags);
  }
  appender.add_iterator_definition(d_selected_out);
  appender.add_iterator_definition(d_num_selected_out);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({compact_init_kernel_name})
      ->add_expression({select_if_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({compact_init_kernel_name, compact_init_kernel_lowered_name})
      ->get_name({select_if_kernel_name, select_if_kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(
    cuLibraryGetKernel(&build_ptr->compact_init_kernel, build_ptr->library, compact_init_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->select_if_kernel, build_ptr->library, select_if_kernel_lowered_name.c_str()));

  nlohmann::json runtime_policy = cub::detail::ptx_json::parse(policy_json_id.c_str(), {result.data.get(), result.size});

  using cub::detail::RuntimeSelectIfAgentPolicy;
  auto select_if_policy = RuntimeSelectIfAgentPolicy::from_json(runtime_policy["Policy"], "SelectIfPolicyT");
  auto vsmem_per_block  = static_cast<::cuda::std::size_t>(runtime_policy["VSMemPerBlock"].get<int>());

  build_ptr->cc             = cc;
  build_ptr->cubin          = (void*) result.data.release();
  build_ptr->cubin_size     = result.size;
  build_ptr->runtime_policy = new select_if::select_if_runtime_tuning_policy{select_if_policy, vsmem_per_block};

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_%s_build(): %s\n", api_name_v<UsesFlags>, exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

template <bool UsesFlags>
CUresult run(cccl_device_select_if_build_result_t build,
             void* d_temp_storage,
             size_t* temp_storage_bytes,
             cccl_iterator_t d_in,
             cccl_iterator_t d_flags,
             cccl_iterator_t d_selected_out,
             cccl_iterator_t d_num_selected_out,
             cccl_op_t select_op,
             uint64_t num_items,
             CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::DispatchSelectIf<
      indirect_arg_t,
      dispatch_flags_iterator_t<UsesFlags>,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      cub::NullType,
      OffsetT,
      cub::SelectImpl::Select,
      select_if::select_if_runtime_tuning_policy,
      select_if::select_if_kernel_source,
      cub::detail::CudaDriverLauncherFactory,
      select_if::dynamic_vsmem_helper_t>::Dispatch(
      d_temp_storage,
      *temp_storage_bytes,
      d_in,
      make_dispatch_flags_iterator<UsesFlags>(d_flags),
      d_selected_out,
      d_num_selected_out,
      select_op,
      cub::NullType{},
      static_cast<OffsetT>(num_items),
      stream,
      {build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
      *reinterpret_cast<select_if::select_if_runtime_tuning_policy*>(build.runtime_policy));

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_%s(): %s\n", api_name_v<UsesFlags>, exc.what());
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

CUresult cleanup(cccl_device_select_if_build_result_t* bld_ptr, const char* api_name)
try
{
  if (bld_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(bld_ptr->cubin));
  std::unique_ptr<select_if::select_if_runtime_tuning_policy> policy(
    reinterpret_cast<select_if::select_if_runtime_tuning_policy*>(bld_ptr->runtime_policy));
  check(cuLibraryUnload(bld_ptr->library));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in %s(): %s\n", api_name, exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}
} // namespace select_if

CUresult cccl_device_select_if_build_ex(
  cccl_device_select_if_build_result_t* build_ptr,
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
  cccl_build_config* config)
{
  return select_if::build_ex<false>(
    build_ptr,
    d_in,
    {},
    d_selected_out,
    d_num_selected_out,
    select_op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    config);
}

CUresult cccl_device_select_flagged_if_build_ex(
  cccl_device_select_if_build_result_t* build_ptr,
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
  cccl_build_config* config)
{
  return select_if::build_ex<true>(
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
    ctk_path,
    config);
}

CUresult cccl_device_select_if(
  cccl_device_select_if_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  uint64_t num_items,
  CUstream stream)
{
  return select_if::run<false>(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    {},
    d_selected_out,
    d_num_selected_out,
    select_op,
    num_items,
    stream);
}

CUresult cccl_device_select_flagged_if(
  cccl_device_select_if_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_flags,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  uint64_t num_items,
  CUstream stream)
{
  return select_if::run<true>(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_flags,
    d_selected_out,
    d_num_selected_out,
    select_op,
    num_items,
    stream);
}

CUresult cccl_device_select_if_cleanup(cccl_device_select_if_build_result_t* bld_ptr)
{
  return select_if::cleanup(bld_ptr, "cccl_device_select_if_cleanup");
}

CUresult cccl_device_select_flagged_if_cleanup(cccl_device_select_if_build_result_t* bld_ptr)
{
  return select_if::cleanup(bld_ptr, "cccl_device_select_flagged_if_cleanup");
}

CUresult cccl_device_select_if_build(
  cccl_device_select_if_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_selected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_select_if_build_ex(
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
    ctk_path,
    nullptr);
}

CUresult cccl_device_select_flagged_if_build(
  cccl_device_select_if_build_result_t* build_ptr,
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
  const char* ctk_path)
{
  return cccl_device_select_flagged_if_build_ex(
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
    ctk_path,
    nullptr);
}
