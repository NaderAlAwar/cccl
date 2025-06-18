//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/device/dispatch/tuning/tuning_transform.cuh> // cub::detail::transform::Algorithm
#include <cub/util_arch.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

// #include <format>
#include <string>
#include <type_traits>

#include "kernels/iterators.h"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/transform.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <stdio.h> // printf

struct op_wrapper;
struct device_transform_policy;

using OffsetT = int;
static_assert(std::is_same_v<cub::detail::choose_signed_offset_t<OffsetT>, OffsetT>,
              "OffsetT must be signed int32 or int64");

struct input_storage_t;
struct input1_storage_t;
struct input2_storage_t;
struct output_storage_t;

namespace transform
{

constexpr auto input_iterator_name  = "input_iterator_t";
constexpr auto input1_iterator_name = "input1_iterator_t";
constexpr auto input2_iterator_name = "input2_iterator_t";
constexpr auto output_iterator_name = "output_iterator_t";

struct transform_runtime_tuning_policy
{
  int block_threads;
  int items_per_thread_no_input;
  int min_items_per_thread;
  int max_items_per_thread;
  int items_per_thread_vectorized;
  int vector_load_length;
  int items_per_thread;
  int load_store_word_size;
  int min_bif;

  // Note: when we extend transform to support UBLKCP, we may no longer
  // be able to keep this constexpr:
  static constexpr cub::detail::transform::Algorithm GetAlgorithm()
  {
    // return cub::detail::transform::Algorithm::prefetch;
    // return cub::detail::transform::Algorithm::vectorized;
    return cub::detail::transform::Algorithm::memcpy_async;
  }

  int BlockThreads()
  {
    return block_threads;
  }

  int ItemsPerThreadNoInput()
  {
    return items_per_thread_no_input;
  }

  int ItemsPerThreadVectorized()
  {
    return items_per_thread_vectorized;
  }

  int MinItemsPerThread()
  {
    return min_items_per_thread;
  }

  int MaxItemsPerThread()
  {
    return max_items_per_thread;
  }

  int VectorLoadLength()
  {
    return vector_load_length;
  }

  int ItemsPerThread()
  {
    return items_per_thread;
  }

  int LoadStoreWordSize()
  {
    return load_store_word_size;
  }

  int MinBif()
  {
    return min_bif;
  }

  // static constexpr int min_bif = 1024 * 16;
};

transform_runtime_tuning_policy get_policy(int cc, int output_size)
{
  // return prefetch policy defaults:
  constexpr int load_store_word_size = 8;
  const int value_type_size          = ::cuda::std::max(output_size, 1);
  const int bytes_per_tile           = ::cuda::round_up(32, ::cuda::std::lcm(load_store_word_size, value_type_size));
  assert(bytes_per_tile % value_type_size == 0);
  const int items_per_thread = bytes_per_tile / value_type_size;
  assert((items_per_thread * value_type_size) % load_store_word_size == 0);
  int min_bif = cub::detail::transform::arch_to_min_bytes_in_flight(cc * 10);
  return {256, 2, 1, 32, 16, 4, items_per_thread, load_store_word_size, min_bif};
}

template <typename StorageT>
const std::string get_iterator_name(cccl_iterator_t iterator, const std::string& name)
{
  if (iterator.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return cccl_type_enum_to_name<StorageT>(iterator.value_type.type, true);
  }
  return name;
}

std::string get_kernel_name(cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t /*op*/)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_transform_policy>(&chained_policy_t));

  const std::string input_iterator_t  = get_iterator_name<input_storage_t>(input_it, input_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&transform_op_t));

  return "cub::detail::transform::transform_kernel<" + chained_policy_t + ", " + offset_t + ", " + transform_op_t + ", "
       + output_iterator_t + ", " + input_iterator_t + ">";
}

std::string
get_kernel_name(cccl_iterator_t input1_it, cccl_iterator_t input2_it, cccl_iterator_t output_it, cccl_op_t /*op*/)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_transform_policy>(&chained_policy_t));

  const std::string input1_iterator_t = get_iterator_name<input1_storage_t>(input1_it, input1_iterator_name);
  const std::string input2_iterator_t = get_iterator_name<input2_storage_t>(input2_it, input2_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&transform_op_t));

  return "cub::detail::transform::transform_kernel<" + chained_policy_t + ", " + offset_t + ", " + "::cuda::std::plus<>"
       + ", " + output_iterator_t + ", " + input1_iterator_t + ", " + input2_iterator_t + ">";
}

template <auto* GetPolicy>
struct dynamic_transform_policy_t
{
  using max_policy = dynamic_transform_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<transform_runtime_tuning_policy>(GetPolicy(device_ptx_version, output_size));
  }

  int output_size;
};

struct transform_kernel_source
{
  cccl_device_transform_build_result_t& build;
  std::vector<cuda::std::size_t> it_value_sizes;
  std::vector<cuda::std::size_t> it_value_alignments;
  cccl_type_enum type;
  bool binary_transform;

  CUkernel TransformKernel() const
  {
    return build.transform_kernel;
  }

  int LoadedBytesPerIteration()
  {
    return build.loaded_bytes_per_iteration;
  }

  template <typename It>
  constexpr It MakeIteratorKernelArg(It it)
  {
    return it;
  }

  static constexpr bool CanVectorize()
  {
    // Here we just need to check that the data type is a primitive and this is a binary transform
    // return type != CCCL_STORAGE;
    return true;
  }

  auto ItValueSizes() const
  {
    return cuda::std::span(it_value_sizes);
  }

  auto ItValueAlignments() const
  {
    return cuda::std::span(it_value_alignments);
  }

  cub::detail::transform::kernel_arg<char*> MakeAlignedBasePtrKernelArg(indirect_arg_t it, int align) const
  {
    return cub::detail::transform::make_aligned_base_ptr_kernel_arg(*static_cast<char**>(&it), align);
  }
};

} // namespace transform

CUresult cccl_device_unary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
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

    const int cc                 = cc_major * 10 + cc_minor;
    const auto policy            = transform::get_policy(cc, output_it.value_type.size);
    const auto input_it_value_t  = cccl_type_enum_to_name<input_storage_t>(input_it.value_type.type);
    const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT32);
    const std::string input_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input_iterator_name, input_it_value_t, input_it);
    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
    const std::string op_src = make_kernel_user_unary_operator(input_it_value_t, output_it_value_t, op);

    [[maybe_unused]] constexpr std::string_view src_template = R"XXX(
#define _CUB_HAS_TRANSFORM_UBLKCP 0
#include <cub/device/dispatch/kernels/transform.cuh>
struct __align__({1}) input_storage_t {{
  char data[{0}];
}};
struct __align__({3}) output_storage_t {{
  char data[{2}];
}};
{8}
{9}
struct vectorized_policy_t {{
  static constexpr int block_threads = {4};
  static constexpr int items_per_thread_no_input        = {5};
  static constexpr int min_items_per_thread             = {6};
  static constexpr int max_items_per_thread             = {7};
  static constexpr int items_per_thread_vectorized      = {8};
  static constexpr int vector_load_length               = {9};
}};
struct device_transform_policy {{
  struct ActivePolicy {{
    static constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;
    using algo_policy = vectorized_policy_t;
  }};
}};
{10}
)XXX";

    const std::string src =
      "#define _CUB_HAS_TRANSFORM_UBLKCP 0\n"
      "#include <cub/device/dispatch/kernels/transform.cuh>\n"
      "struct __align__("
      + std::to_string(input_it.value_type.alignment)
      + ") input_storage_t {\n"
        "  char data["
      + std::to_string(input_it.value_type.size)
      + "];\n"
        "};\n"
        "struct __align__("
      + std::to_string(output_it.value_type.alignment)
      + ") output_storage_t {\n"
        "  char data["
      + std::to_string(output_it.value_type.size)
      + "];\n"
        "};\n"
      + input_iterator_src + "\n" + output_iterator_src
      + "\n"
        "struct vectorized_policy_t {\n"
        "  static constexpr int block_threads = "
      + std::to_string(policy.block_threads)
      + ";\n"
        "  static constexpr int items_per_thread_no_input = "
      + std::to_string(policy.items_per_thread_no_input)
      + ";\n"
        "  static constexpr int min_items_per_thread = "
      + std::to_string(policy.min_items_per_thread)
      + ";\n"
        "  static constexpr int max_items_per_thread = "
      + std::to_string(policy.max_items_per_thread)
      + ";\n"
        "  static constexpr int items_per_thread_vectorized = "
      + std::to_string(policy.items_per_thread_vectorized)
      + ";\n"
        "  static constexpr int vector_load_length = "
      + std::to_string(policy.vector_load_length)
      + ";\n"
        "  static constexpr int items_per_thread = "
      + std::to_string(policy.items_per_thread)
      + ";\n"
        "  static constexpr int load_store_word_size = "
      + std::to_string(policy.load_store_word_size)
      + ";};\n"
        "struct device_transform_policy {\n"
        "  struct ActivePolicy {\n"
        // "    static constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;\n"
        // "    static constexpr auto algorithm = cub::detail::transform::Algorithm::vectorized;\n"
        "    static constexpr auto algorithm = cub::detail::transform::Algorithm::memcpy_async;\n"
        "    using algo_policy = vectorized_policy_t;\n"
        "  };\n"
        "};\n"
      + op_src + "\n";

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string kernel_name = transform::get_kernel_name(input_it, output_it, op);
    std::string kernel_lowered_name;

    const std::string arch = "-arch=sm_" + std::to_string(cc_major) + std::to_string(cc_minor);

    // Note: `-default-device` is needed because of the use of lambdas
    // in the transform kernel code. Qualifying those explicitly with
    // `__device__` seems not to be supported by NVRTC.
    constexpr size_t num_args  = 9;
    const char* args[num_args] = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-default-device",
      "-DCUB_DISABLE_CDP"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};

    appender.append({op.ltoir, op.ltoir_size});
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);

    nvrtc_link_result result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({kernel_name})
        .compile_program({args, num_args})
        .get_name({kernel_name, kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->transform_kernel, build_ptr->library, kernel_lowered_name.c_str()));

    build_ptr->loaded_bytes_per_iteration = input_it.value_type.size;
    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unary_transform_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CUresult cccl_device_unary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));
    auto cuda_error = cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no, // TODO implement yes
      OffsetT,
      ::cuda::std::tuple<indirect_arg_t>,
      indirect_arg_t,
      indirect_arg_t,
      transform::dynamic_transform_policy_t<&transform::get_policy>,
      transform::transform_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      dispatch(d_in,
               d_out,
               num_items,
               op,
               stream,
               {build, {d_in.value_type.size}, {d_in.value_type.alignment}, d_out.value_type.type, false},
               cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
               {static_cast<int>(d_out.value_type.size)});
    if (cuda_error != cudaSuccess)
    {
      const char* errorString = cudaGetErrorString(cuda_error); // Get the error string
      std::cerr << "CUDA error: " << errorString << std::endl;
    }
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unary_transform(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

CUresult cccl_device_binary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t input1_it,
  cccl_iterator_t input2_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
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

    const int cc                 = cc_major * 10 + cc_minor;
    const auto policy            = transform::get_policy(cc, output_it.value_type.size);
    const auto input1_it_value_t = cccl_type_enum_to_name<input1_storage_t>(input1_it.value_type.type);
    const auto input2_it_value_t = cccl_type_enum_to_name<input2_storage_t>(input2_it.value_type.type);

    const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT32);
    const std::string input1_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input1_iterator_name, input1_it_value_t, input1_it);
    const std::string input2_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input2_iterator_name, input2_it_value_t, input2_it);

    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
    const std::string op_src =
      make_kernel_user_binary_operator(input1_it_value_t, input2_it_value_t, output_it_value_t, op);

    [[maybe_unused]] constexpr std::string_view src_template = R"XXX(
#define _CUB_HAS_TRANSFORM_UBLKCP 0
#include <cub/device/dispatch/kernels/transform.cuh>
struct __align__({1}) input1_storage_t {{
  char data[{0}];
}};
struct __align__({3}) input2_storage_t {{
  char data[{2}];
}};

struct __align__({5}) output_storage_t {{
  char data[{4}];
}};

{10}
{11}
{12}

struct prefetch_policy_t {{
  static constexpr int block_threads = {6};
//  static constexpr int items_per_thread_no_input = {7};
  static constexpr int min_items_per_thread      = {8};
  static constexpr int max_items_per_thread      = {9};
}};

struct device_transform_policy {{
  struct ActivePolicy {{
    static constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;
    using algo_policy = prefetch_policy_t;
  }};
}};

{13}
)XXX";
    const std::string src =
      "#define _CUB_HAS_TRANSFORM_UBLKCP 0\n"
      "#include <cub/device/dispatch/kernels/transform.cuh>\n"
      "struct __align__("
      + std::to_string(input1_it.value_type.alignment)
      + ") input1_storage_t {\n"
        "  char data["
      + std::to_string(input1_it.value_type.size)
      + "];\n"
        "};\n"
        "struct __align__("
      + std::to_string(input2_it.value_type.alignment)
      + ") input2_storage_t {\n"
        "  char data["
      + std::to_string(input2_it.value_type.size)
      + "];\n"
        "};\n"
        "struct __align__("
      + std::to_string(output_it.value_type.alignment)
      + ") output_storage_t {\n"
        "  char data["
      + std::to_string(output_it.value_type.size)
      + "];\n"
        "};\n"
      + input1_iterator_src + "\n" + input2_iterator_src + "\n" + output_iterator_src
      + "\n"
        "struct vectorized_policy_t {\n"
        "  static constexpr int block_threads = "
      + std::to_string(policy.block_threads)
      + ";\n"
        "  static constexpr int items_per_thread_no_input = "
      + std::to_string(policy.items_per_thread_no_input)
      + ";\n"
        "  static constexpr int min_items_per_thread = "
      + std::to_string(policy.min_items_per_thread)
      + ";\n"
        "  static constexpr int max_items_per_thread = "
      + std::to_string(policy.max_items_per_thread)
      + ";\n"
        "  static constexpr int items_per_thread_vectorized = "
      + std::to_string(policy.items_per_thread_vectorized)
      + ";\n"
        "  static constexpr int vector_load_length = "
      + std::to_string(policy.vector_load_length)
      + ";\n"
        "  static constexpr int items_per_thread = "
      + std::to_string(policy.items_per_thread)
      + ";\n"
        "  static constexpr int load_store_word_size = "
      + std::to_string(policy.load_store_word_size)
      + ";};\n"
        "struct device_transform_policy {\n"
        "  struct ActivePolicy {\n"
        // "    static constexpr auto algorithm = cub::detail::transform::Algorithm::vectorized;\n"
        // "    static constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;\n"
        "    static constexpr auto algorithm = cub::detail::transform::Algorithm::memcpy_async;\n"
        "    using algo_policy = vectorized_policy_t;\n"
        "  };\n"
        "};\n";
    // + op_src + "\n";

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string kernel_name = transform::get_kernel_name(input1_it, input2_it, output_it, op);
    std::string kernel_lowered_name;

    const std::string arch = "-arch=sm_" + std::to_string(cc_major) + std::to_string(cc_minor);

    constexpr size_t num_args  = 8;
    const char* args[num_args] = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-default-device"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};

    appender.append({op.ltoir, op.ltoir_size});
    appender.add_iterator_definition(input1_it);
    appender.add_iterator_definition(input2_it);
    appender.add_iterator_definition(output_it);

    nvrtc_link_result result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({kernel_name})
        .compile_program({args, num_args})
        .get_name({kernel_name, kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->transform_kernel, build_ptr->library, kernel_lowered_name.c_str()));

    build_ptr->loaded_bytes_per_iteration = (input1_it.value_type.size + input2_it.value_type.size);
    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_binary_transform_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CUresult cccl_device_binary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no, // TODO implement yes
      OffsetT,
      ::cuda::std::tuple<indirect_arg_t, indirect_arg_t>,
      indirect_arg_t,
      indirect_arg_t,
      transform::dynamic_transform_policy_t<&transform::get_policy>,
      transform::transform_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      dispatch(
        ::cuda::std::make_tuple<indirect_arg_t, indirect_arg_t>(d_in1, d_in2),
        d_out,
        num_items,
        op,
        stream,
        {build,
         {d_in1.value_type.size, d_in2.value_type.size},
         {d_in1.value_type.alignment, d_in2.value_type.alignment},
         d_out.value_type.type,
         true},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        {static_cast<int>(d_out.value_type.size)});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_binary_transform(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

CUresult cccl_device_transform_cleanup(cccl_device_transform_build_result_t* build_ptr)
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
    printf("\nEXCEPTION in cccl_device_transform_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
