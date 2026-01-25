// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_partition.cuh>
#include <cub/device/device_select.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cuda/std/algorithm>
#include <cuda/std/functional>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  else // TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename InputT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using SelectIfPolicyT =
      cub::AgentSelectIfPolicy<TUNE_THREADS_PER_BLOCK,
                               TUNE_ITEMS_PER_THREAD,
                               TUNE_LOAD_ALGORITHM,
                               TUNE_LOAD_MODIFIER,
                               cub::BLOCK_SCAN_WARP_SCANS,
                               delay_constructor_t>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <class T>
struct is_nonzero_t
{
  [[nodiscard]] __device__ bool operator()(const T& val) const noexcept
  {
    return val != T{};
  }
};

struct select_mask_tuple_t
{
  template <class Tuple>
  [[nodiscard]] __device__ bool operator()(const Tuple& t) const noexcept
  {
    return thrust::get<1>(t) != 0;
  }
};

struct extract_tuple_value_t
{
  template <class Tuple>
  [[nodiscard]] __device__ auto operator()(const Tuple& t) const noexcept
  {
    return thrust::get<0>(t);
  }
};

template <typename T, typename OffsetT, typename InPlace>
void select(nvbench::state& state, nvbench::type_list<T, OffsetT, InPlace>)
{
  using value_it_t        = const T*;
  using mask_it_t         = const int32_t*;
  using input_it_t        = thrust::zip_iterator<thrust::tuple<value_it_t, mask_it_t>>;
  using flag_it_t         = cub::NullType*;
  using output_it_t       = thrust::transform_output_iterator<extract_tuple_value_t, T*>;
  using num_selected_it_t = OffsetT*;
  using select_op_t       = select_mask_tuple_t;
  using equality_op_t     = cub::NullType;
  using offset_t          = OffsetT;
  constexpr cub::SelectImpl selection_option =
    InPlace::value ? cub::SelectImpl::SelectPotentiallyInPlace : cub::SelectImpl::Select;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchSelectIf<
    input_it_t,
    flag_it_t,
    output_it_t,
    num_selected_it_t,
    select_op_t,
    equality_op_t,
    offset_t,
    selection_option,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchSelectIf<input_it_t,
                          flag_it_t,
                          output_it_t,
                          num_selected_it_t,
                          select_op_t,
                          equality_op_t,
                          offset_t,
                          selection_option>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  select_op_t select_op{};

  thrust::device_vector<T> in         = generate(elements, entropy, T{0}, T{1});
  thrust::device_vector<int32_t> mask = generate(elements, entropy, int32_t{0}, int32_t{1});
  thrust::device_vector<offset_t> num_selected(1);

  // TODO Extract into helper TU
  const auto selected_elements = thrust::count_if(
    thrust::make_zip_iterator(thrust::make_tuple(in.cbegin(), mask.cbegin())),
    thrust::make_zip_iterator(thrust::make_tuple(in.cend(), mask.cend())),
    select_op);

  thrust::device_vector<T> out;
  value_it_t d_in  = thrust::raw_pointer_cast(in.data());
  mask_it_t d_mask = thrust::raw_pointer_cast(mask.data());
  output_it_t d_out =
    thrust::make_transform_output_iterator(thrust::raw_pointer_cast(in.data()), extract_tuple_value_t{});
  if constexpr (!InPlace::value)
  {
    out   = thrust::device_vector<T>(selected_elements);
    d_out = thrust::make_transform_output_iterator(thrust::raw_pointer_cast(out.data()), extract_tuple_value_t{});
  }

  flag_it_t d_flags                = nullptr;
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<int32_t>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<offset_t>(1);

  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::make_zip_iterator(thrust::make_tuple(d_in, d_mask)),
    d_flags,
    d_out,
    d_num_selected,
    select_op,
    equality_op_t{},
    elements,
    0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      thrust::make_zip_iterator(thrust::make_tuple(d_in, d_mask)),
      d_flags,
      d_out,
      d_num_selected,
      select_op,
      equality_op_t{},
      elements,
      launch.get_stream());
  });
}

template <class T>
struct always_false_t
{
  [[nodiscard]] __device__ bool operator()(const T&) const noexcept
  {
    return false;
  }
};

template <typename T, typename OffsetT>
void select_three_way(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using value_it_t        = const T*;
  using mask_it_t         = const int32_t*;
  using input_it_t        = thrust::zip_iterator<thrust::tuple<value_it_t, mask_it_t>>;
  using output_it_t       = thrust::transform_output_iterator<extract_tuple_value_t, T*>;
  using discard_it_t      = thrust::discard_iterator<>;
  using num_selected_it_t = OffsetT*;
  using select_op_t       = select_mask_tuple_t;
  using reject_op_t       = always_false_t<thrust::tuple<T, int32_t>>;
  using offset_t          = OffsetT;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchThreeWayPartitionIf<
    input_it_t,
    output_it_t,
    discard_it_t,
    discard_it_t,
    num_selected_it_t,
    select_op_t,
    reject_op_t,
    offset_t,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::DispatchThreeWayPartitionIf<
    input_it_t,
    output_it_t,
    discard_it_t,
    discard_it_t,
    num_selected_it_t,
    select_op_t,
    reject_op_t,
    offset_t>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  select_op_t select_op{};
  reject_op_t reject_op{};

  thrust::device_vector<T> in         = generate(elements, entropy, T{0}, T{1});
  thrust::device_vector<int32_t> mask = generate(elements, entropy, int32_t{0}, int32_t{1});

  // TODO Extract into helper TU
  const auto selected_elements = thrust::count_if(
    thrust::make_zip_iterator(thrust::make_tuple(in.cbegin(), mask.cbegin())),
    thrust::make_zip_iterator(thrust::make_tuple(in.cend(), mask.cend())),
    select_op);
  thrust::device_vector<T> out(selected_elements);
  thrust::device_vector<offset_t> num_selected(2);

  value_it_t d_in  = thrust::raw_pointer_cast(in.data());
  mask_it_t d_mask = thrust::raw_pointer_cast(mask.data());
  output_it_t d_out =
    thrust::make_transform_output_iterator(thrust::raw_pointer_cast(out.data()), extract_tuple_value_t{});
  discard_it_t d_out_2{};
  discard_it_t d_out_3{};
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<int32_t>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<offset_t>(2);

  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::make_zip_iterator(thrust::make_tuple(d_in, d_mask)),
    d_out,
    d_out_2,
    d_out_3,
    d_num_selected,
    select_op,
    reject_op,
    elements,
    0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      thrust::make_zip_iterator(thrust::make_tuple(d_in, d_mask)),
      d_out,
      d_out_2,
      d_out_3,
      d_num_selected,
      select_op,
      reject_op,
      elements,
      launch.get_stream());
  });
}

template <typename T, typename OffsetT>
void select_copy_if(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t  = const T*;
  using mask_it_t   = const int32_t*;
  using output_it_t = T*;

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<T> in         = generate(elements, entropy, T{0}, T{1});
  thrust::device_vector<int32_t> mask = generate(elements, entropy, int32_t{0}, int32_t{1});
  thrust::device_vector<T> out(elements);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  mask_it_t d_mask  = thrust::raw_pointer_cast(mask.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<int32_t>(elements);
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               auto policy = thrust::cuda::par.on(launch.get_stream());
               thrust::copy_if(policy, d_in, d_in + elements, d_mask, d_out, ::cuda::std::identity{});
             });
}

// Benchmark that directly calls cub::DispatchSelectIf with separate input/stencil iterators
// (same pattern as thrust::copy_if uses internally)
template <typename T, typename OffsetT>
void select_cub_stencil(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t                           = const T*;
  using stencil_it_t                         = const int32_t*;
  using output_it_t                          = T*;
  using num_selected_it_t                    = OffsetT*;
  using select_op_t                          = ::cuda::std::identity;
  using equality_op_t                        = cub::NullType;
  using offset_t                             = OffsetT;
  constexpr cub::SelectImpl selection_option = cub::SelectImpl::Select;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchSelectIf<
    input_it_t,
    stencil_it_t,
    output_it_t,
    num_selected_it_t,
    select_op_t,
    equality_op_t,
    offset_t,
    selection_option,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchSelectIf<input_it_t,
                          stencil_it_t,
                          output_it_t,
                          num_selected_it_t,
                          select_op_t,
                          equality_op_t,
                          offset_t,
                          selection_option>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  select_op_t select_op{};

  thrust::device_vector<T> in            = generate(elements, entropy, T{0}, T{1});
  thrust::device_vector<int32_t> stencil = generate(elements, entropy, int32_t{0}, int32_t{1});
  thrust::device_vector<T> out(elements);
  thrust::device_vector<offset_t> num_selected(1);

  input_it_t d_in                  = thrust::raw_pointer_cast(in.data());
  stencil_it_t d_stencil           = thrust::raw_pointer_cast(stencil.data());
  output_it_t d_out                = thrust::raw_pointer_cast(out.data());
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<int32_t>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_writes<offset_t>(1);

  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_stencil, d_out, d_num_selected, select_op, equality_op_t{}, elements, 0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_stencil,
      d_out,
      d_num_selected,
      select_op,
      equality_op_t{},
      elements,
      launch.get_stream());
  });
}

// Benchmark that mimics parrot's where() pattern:
// - Input: CountingIterator (indices generated on-the-fly, no memory read)
// - Stencil: mask array (read from memory)
// - Output: selected indices
// This has ~40% less memory traffic than the standard benchmark
template <typename T, typename OffsetT>
void select_counting_iterator(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t                           = thrust::counting_iterator<T>;
  using stencil_it_t                         = const int32_t*;
  using output_it_t                          = T*;
  using num_selected_it_t                    = OffsetT*;
  using select_op_t                          = ::cuda::std::identity;
  using equality_op_t                        = cub::NullType;
  using offset_t                             = OffsetT;
  constexpr cub::SelectImpl selection_option = cub::SelectImpl::Select;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchSelectIf<
    input_it_t,
    stencil_it_t,
    output_it_t,
    num_selected_it_t,
    select_op_t,
    equality_op_t,
    offset_t,
    selection_option,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t =
    cub::DispatchSelectIf<input_it_t,
                          stencil_it_t,
                          output_it_t,
                          num_selected_it_t,
                          select_op_t,
                          equality_op_t,
                          offset_t,
                          selection_option>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  select_op_t select_op{};

  // Input is a counting iterator - no memory allocation needed!
  input_it_t d_in(0); // CountingIterator starting at 0

  // Stencil/mask is still a real array
  thrust::device_vector<int32_t> stencil = generate(elements, entropy, int32_t{0}, int32_t{1});
  stencil_it_t d_stencil                 = thrust::raw_pointer_cast(stencil.data());

  // Count selected elements for output allocation
  const auto selected_elements = thrust::count_if(stencil.begin(), stencil.end(), select_op);

  thrust::device_vector<T> out(selected_elements);
  thrust::device_vector<offset_t> num_selected(1);

  output_it_t d_out                = thrust::raw_pointer_cast(out.data());
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  // Only counting stencil reads (no input reads!) and output writes
  state.add_global_memory_reads<int32_t>(elements); // stencil only
  state.add_global_memory_writes<T>(selected_elements);
  state.add_global_memory_writes<offset_t>(1);

  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_stencil, d_out, d_num_selected, select_op, equality_op_t{}, elements, 0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_stencil,
      d_out,
      d_num_selected,
      select_op,
      equality_op_t{},
      elements,
      launch.get_stream());
  });
}

// Benchmark three-way partition with CountingIterator input (like parrot's where())
// - Input: zip(CountingIterator, mask) - indices generated on-the-fly
// - Output: selected indices where mask is non-zero
// This has ~40% less memory traffic than the standard three_way benchmark
template <typename T, typename OffsetT>
void select_three_way_counting_iterator(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using count_it_t        = thrust::counting_iterator<T>;
  using mask_it_t         = const int32_t*;
  using input_it_t        = thrust::zip_iterator<thrust::tuple<count_it_t, mask_it_t>>;
  using output_it_t       = thrust::transform_output_iterator<extract_tuple_value_t, T*>;
  using discard_it_t      = thrust::discard_iterator<>;
  using num_selected_it_t = OffsetT*;
  using select_op_t       = select_mask_tuple_t;
  using reject_op_t       = always_false_t<thrust::tuple<T, int32_t>>;
  using offset_t          = OffsetT;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<T>;
  using dispatch_t = cub::DispatchThreeWayPartitionIf<
    input_it_t,
    output_it_t,
    discard_it_t,
    discard_it_t,
    num_selected_it_t,
    select_op_t,
    reject_op_t,
    offset_t,
    policy_t>;
#else // TUNE_BASE
  using dispatch_t = cub::DispatchThreeWayPartitionIf<
    input_it_t,
    output_it_t,
    discard_it_t,
    discard_it_t,
    num_selected_it_t,
    select_op_t,
    reject_op_t,
    offset_t>;
#endif // TUNE_BASE

  // Retrieve axis parameters
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  select_op_t select_op{};
  reject_op_t reject_op{};

  // Input is a counting iterator zipped with mask
  count_it_t d_count(0); // CountingIterator starting at 0

  // Mask is a real array
  thrust::device_vector<int32_t> mask = generate(elements, entropy, int32_t{0}, int32_t{1});
  mask_it_t d_mask                    = thrust::raw_pointer_cast(mask.data());

  // Count selected for output allocation
  const auto selected_elements = thrust::count_if(
    thrust::make_zip_iterator(thrust::make_tuple(d_count, mask.cbegin())),
    thrust::make_zip_iterator(thrust::make_tuple(d_count + elements, mask.cend())),
    select_op);

  thrust::device_vector<T> out(selected_elements);
  thrust::device_vector<offset_t> num_selected(2);

  output_it_t d_out =
    thrust::make_transform_output_iterator(thrust::raw_pointer_cast(out.data()), extract_tuple_value_t{});
  discard_it_t d_out_2{};
  discard_it_t d_out_3{};
  num_selected_it_t d_num_selected = thrust::raw_pointer_cast(num_selected.data());

  state.add_element_count(elements);
  // Only counting mask reads (no value reads!) and output writes
  state.add_global_memory_reads<int32_t>(elements); // mask only
  state.add_global_memory_writes<T>(selected_elements);
  state.add_global_memory_writes<offset_t>(2);

  std::size_t temp_size{};
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::make_zip_iterator(thrust::make_tuple(d_count, d_mask)),
    d_out,
    d_out_2,
    d_out_3,
    d_num_selected,
    select_op,
    reject_op,
    elements,
    0);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      thrust::make_zip_iterator(thrust::make_tuple(d_count, d_mask)),
      d_out,
      d_out_2,
      d_out_3,
      d_num_selected,
      select_op,
      reject_op,
      elements,
      launch.get_stream());
  });
}

// The implementation of DeviceSelect for 64-bit offset types uses a streaming approach, where it runs multiple passes
// using a 32-bit offset type, so we only need to test one (to save time for tuning and the benchmark CI).
using select_offset_types = nvbench::type_list<int32_t, int64_t>;

using ::cuda::std::false_type;
using ::cuda::std::true_type;

using my_types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(select, NVBENCH_TYPE_AXES(my_types, select_offset_types, nvbench::type_list<false_type>))
  .set_name("out_of_place")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "InPlace{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});

NVBENCH_BENCH_TYPES(select, NVBENCH_TYPE_AXES(my_types, select_offset_types, nvbench::type_list<true_type>))
  .set_name("in_place")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "InPlace{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});

NVBENCH_BENCH_TYPES(select_three_way, NVBENCH_TYPE_AXES(my_types, select_offset_types))
  .set_name("three_way_select_like")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});

NVBENCH_BENCH_TYPES(select_copy_if, NVBENCH_TYPE_AXES(my_types, select_offset_types))
  .set_name("thrust_copy_if")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});

NVBENCH_BENCH_TYPES(select_cub_stencil, NVBENCH_TYPE_AXES(my_types, select_offset_types))
  .set_name("cub_stencil_select")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});

NVBENCH_BENCH_TYPES(select_counting_iterator, NVBENCH_TYPE_AXES(my_types, select_offset_types))
  .set_name("counting_iterator_select")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});

NVBENCH_BENCH_TYPES(select_three_way_counting_iterator, NVBENCH_TYPE_AXES(my_types, select_offset_types))
  .set_name("three_way_counting_iterator")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Elements{io}", {100000000})
  .add_string_axis("Entropy", {"1.000"});
