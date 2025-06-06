#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/partition.h>

#include "thrust/detail/raw_pointer_cast.h"
#include <unittest/unittest.h>

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x) const
  {
    return ((int) x % 2) == 0;
  }
};

template <typename T>
struct mod_n
{
  T mod;
  bool negate;
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return (x % mod == 0) ? (!negate) : negate;
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  _CCCL_HOST_DEVICE T operator()(T x)
  {
    return x * multiplier;
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Predicate, typename Iterator2>
__global__ void partition_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result)
{
  *result = thrust::partition(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestPartitionDevice(ExecutionPolicy exec)
{
  using T        = int;
  using iterator = typename thrust::device_vector<T>::iterator;

  thrust::device_vector<T> data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 1;
  data[3] = 1;
  data[4] = 2;

  thrust::device_vector<iterator> result(1);

  partition_kernel<<<1, 1>>>(exec, data.begin(), data.end(), is_even<T>(), result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::device_vector<T> ref(5);
  ref[0] = 2;
  ref[1] = 2;
  ref[2] = 1;
  ref[3] = 1;
  ref[4] = 1;

  ASSERT_EQUAL(2, (iterator) result[0] - data.begin());
  ASSERT_EQUAL(ref, data);
}

void TestPartitionDeviceSeq()
{
  TestPartitionDevice(thrust::seq);
}
DECLARE_UNITTEST(TestPartitionDeviceSeq);

void TestPartitionDeviceDevice()
{
  TestPartitionDevice(thrust::device);
}
DECLARE_UNITTEST(TestPartitionDeviceDevice);

void TestPartitionDeviceNoSync()
{
  TestPartitionDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestPartitionDeviceNoSync);

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void partition_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result)
{
  *result = thrust::partition(exec, first, last, stencil_first, pred);
}

template <typename ExecutionPolicy>
void TestPartitionStencilDevice(ExecutionPolicy exec)
{
  using T        = int;
  using iterator = typename thrust::device_vector<T>::iterator;

  thrust::device_vector<T> data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 0;
  data[3] = 0;
  data[4] = 1;

  thrust::device_vector<T> stencil(5);
  stencil[0] = 1;
  stencil[1] = 2;
  stencil[2] = 1;
  stencil[3] = 1;
  stencil[4] = 2;

  thrust::device_vector<iterator> result(1);

  partition_kernel<<<1, 1>>>(exec, data.begin(), data.end(), stencil.begin(), is_even<T>(), result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::device_vector<T> ref(5);
  ref[0] = 1;
  ref[1] = 1;
  ref[2] = 0;
  ref[3] = 0;
  ref[4] = 0;

  ASSERT_EQUAL(2, (iterator) result[0] - data.begin());
  ASSERT_EQUAL(ref, data);
}

void TestPartitionStencilDeviceSeq()
{
  TestPartitionStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestPartitionStencilDeviceSeq);

void TestPartitionStencilDeviceDevice()
{
  TestPartitionStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestPartitionStencilDeviceDevice);

void TestPartitionStencilDeviceNoSync()
{
  TestPartitionStencilDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestPartitionStencilDeviceNoSync);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Predicate,
          typename Iterator4>
__global__ void partition_copy_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 true_result,
  Iterator3 false_result,
  Predicate pred,
  Iterator4 result)
{
  *result = thrust::partition_copy(exec, first, last, true_result, false_result, pred);
}

template <typename ExecutionPolicy>
void TestPartitionCopyDevice(ExecutionPolicy exec)
{
  using T        = int;
  using iterator = thrust::device_vector<T>::iterator;

  thrust::device_vector<T> data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 1;
  data[3] = 1;
  data[4] = 2;

  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  using pair_type = thrust::pair<iterator, iterator>;
  thrust::device_vector<pair_type> iterators(1);

  partition_copy_kernel<<<1, 1>>>(
    exec, data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>(), iterators.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::device_vector<T> true_ref(2);
  true_ref[0] = 2;
  true_ref[1] = 2;

  thrust::device_vector<T> false_ref(3);
  false_ref[0] = 1;
  false_ref[1] = 1;
  false_ref[2] = 1;

  pair_type ends = iterators[0];

  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}

void TestPartitionCopyDeviceSeq()
{
  TestPartitionCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestPartitionCopyDeviceSeq);

void TestPartitionCopyDeviceDevice()
{
  TestPartitionCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestPartitionCopyDeviceDevice);

void TestPartitionCopyDeviceNoSync()
{
  TestPartitionCopyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestPartitionCopyDeviceNoSync);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Predicate,
          typename Iterator5>
__global__ void partition_copy_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 stencil_first,
  Iterator3 true_result,
  Iterator4 false_result,
  Predicate pred,
  Iterator5 result)
{
  *result = thrust::partition_copy(exec, first, last, stencil_first, true_result, false_result, pred);
}

template <typename ExecutionPolicy>
void TestPartitionCopyStencilDevice(ExecutionPolicy exec)
{
  using T = int;

  thrust::device_vector<int> data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 0;
  data[3] = 0;
  data[4] = 1;

  thrust::device_vector<int> stencil(5);
  stencil[0] = 1;
  stencil[1] = 2;
  stencil[2] = 1;
  stencil[3] = 1;
  stencil[4] = 2;

  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  using iterator  = typename thrust::device_vector<int>::iterator;
  using pair_type = thrust::pair<iterator, iterator>;
  thrust::device_vector<pair_type> iterators(1);

  partition_copy_kernel<<<1, 1>>>(
    exec,
    data.begin(),
    data.end(),
    stencil.begin(),
    true_results.begin(),
    false_results.begin(),
    is_even<T>(),
    iterators.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  pair_type ends = iterators[0];

  thrust::device_vector<int> true_ref(2);
  true_ref[0] = 1;
  true_ref[1] = 1;

  thrust::device_vector<int> false_ref(3);
  false_ref[0] = 0;
  false_ref[1] = 0;
  false_ref[2] = 0;

  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}

void TestPartitionCopyStencilDeviceSeq()
{
  TestPartitionCopyStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestPartitionCopyStencilDeviceSeq);

void TestPartitionCopyStencilDeviceDevice()
{
  TestPartitionCopyStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestPartitionCopyStencilDeviceDevice);

void TestPartitionCopyStencilDeviceNoSync()
{
  TestPartitionCopyStencilDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestPartitionCopyStencilDeviceNoSync);

template <typename ExecutionPolicy, typename Iterator1, typename Predicate, typename Iterator2>
__global__ void
stable_partition_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result)
{
  *result = thrust::stable_partition(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestStablePartitionDevice(ExecutionPolicy exec)
{
  using T        = int;
  using iterator = typename thrust::device_vector<T>::iterator;

  thrust::device_vector<T> data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 1;
  data[3] = 1;
  data[4] = 2;

  thrust::device_vector<iterator> result(1);

  stable_partition_kernel<<<1, 1>>>(exec, data.begin(), data.end(), is_even<T>(), result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::device_vector<T> ref(5);
  ref[0] = 2;
  ref[1] = 2;
  ref[2] = 1;
  ref[3] = 1;
  ref[4] = 1;

  ASSERT_EQUAL(2, (iterator) result[0] - data.begin());
  ASSERT_EQUAL(ref, data);
}

void TestStablePartitionDeviceSeq()
{
  TestStablePartitionDevice(thrust::seq);
}
DECLARE_UNITTEST(TestStablePartitionDeviceSeq);

void TestStablePartitionDeviceDevice()
{
  TestStablePartitionDevice(thrust::device);
}
DECLARE_UNITTEST(TestStablePartitionDeviceDevice);

void TestStablePartitionDeviceNoSync()
{
  TestStablePartitionDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestStablePartitionDeviceNoSync);

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void stable_partition_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result)
{
  *result = thrust::stable_partition(exec, first, last, stencil_first, pred);
}

template <typename ExecutionPolicy>
void TestStablePartitionStencilDevice(ExecutionPolicy exec)
{
  using T        = int;
  using iterator = typename thrust::device_vector<T>::iterator;

  thrust::device_vector<T> data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 0;
  data[3] = 0;
  data[4] = 1;

  thrust::device_vector<T> stencil(5);
  stencil[0] = 1;
  stencil[1] = 2;
  stencil[2] = 1;
  stencil[3] = 1;
  stencil[4] = 2;

  thrust::device_vector<iterator> result(1);

  stable_partition_kernel<<<1, 1>>>(exec, data.begin(), data.end(), stencil.begin(), is_even<T>(), result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::device_vector<T> ref(5);
  ref[0] = 1;
  ref[1] = 1;
  ref[2] = 0;
  ref[3] = 0;
  ref[4] = 0;

  ASSERT_EQUAL(2, (iterator) result[0] - data.begin());
  ASSERT_EQUAL(ref, data);
}

void TestStablePartitionStencilDeviceSeq()
{
  TestStablePartitionStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestStablePartitionStencilDeviceSeq);

void TestStablePartitionStencilDeviceDevice()
{
  TestStablePartitionStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestStablePartitionStencilDeviceDevice);

void TestStablePartitionStencilDeviceNoSync()
{
  TestStablePartitionStencilDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestStablePartitionStencilDeviceNoSync);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Predicate,
          typename Iterator4>
__global__ void stable_partition_copy_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 true_result,
  Iterator3 false_result,
  Predicate pred,
  Iterator4 result)
{
  *result = thrust::stable_partition_copy(exec, first, last, true_result, false_result, pred);
}

template <typename ExecutionPolicy>
void TestStablePartitionCopyDevice(ExecutionPolicy exec)
{
  using T        = int;
  using iterator = thrust::device_vector<T>::iterator;

  thrust::device_vector<T> data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 1;
  data[3] = 1;
  data[4] = 2;

  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  using pair_type = thrust::pair<iterator, iterator>;
  thrust::device_vector<pair_type> iterators(1);

  stable_partition_copy_kernel<<<1, 1>>>(
    exec, data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>(), iterators.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::device_vector<T> true_ref(2);
  true_ref[0] = 2;
  true_ref[1] = 2;

  thrust::device_vector<T> false_ref(3);
  false_ref[0] = 1;
  false_ref[1] = 1;
  false_ref[2] = 1;

  pair_type ends = iterators[0];

  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}

void TestStablePartitionCopyDeviceSeq()
{
  TestStablePartitionCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestStablePartitionCopyDeviceSeq);

void TestStablePartitionCopyDeviceDevice()
{
  TestStablePartitionCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestStablePartitionCopyDeviceDevice);

void TestStablePartitionCopyDeviceNoSync()
{
  TestStablePartitionCopyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestStablePartitionCopyDeviceNoSync);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Predicate,
          typename Iterator5>
__global__ void stable_partition_copy_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 stencil_first,
  Iterator3 true_result,
  Iterator4 false_result,
  Predicate pred,
  Iterator5 result)
{
  *result = thrust::stable_partition_copy(exec, first, last, stencil_first, true_result, false_result, pred);
}

template <typename ExecutionPolicy>
void TestStablePartitionCopyStencilDevice(ExecutionPolicy exec)
{
  using T = int;

  thrust::device_vector<int> data(5);
  data[0] = 0;
  data[1] = 1;
  data[2] = 0;
  data[3] = 0;
  data[4] = 1;

  thrust::device_vector<int> stencil(5);
  stencil[0] = 1;
  stencil[1] = 2;
  stencil[2] = 1;
  stencil[3] = 1;
  stencil[4] = 2;

  thrust::device_vector<int> true_results(2);
  thrust::device_vector<int> false_results(3);

  using iterator  = typename thrust::device_vector<int>::iterator;
  using pair_type = thrust::pair<iterator, iterator>;
  thrust::device_vector<pair_type> iterators(1);

  stable_partition_copy_kernel<<<1, 1>>>(
    exec,
    data.begin(),
    data.end(),
    stencil.begin(),
    true_results.begin(),
    false_results.begin(),
    is_even<T>(),
    iterators.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  pair_type ends = iterators[0];

  thrust::device_vector<int> true_ref(2);
  true_ref[0] = 1;
  true_ref[1] = 1;

  thrust::device_vector<int> false_ref(3);
  false_ref[0] = 0;
  false_ref[1] = 0;
  false_ref[2] = 0;

  ASSERT_EQUAL(2, ends.first - true_results.begin());
  ASSERT_EQUAL(3, ends.second - false_results.begin());
  ASSERT_EQUAL(true_ref, true_results);
  ASSERT_EQUAL(false_ref, false_results);
}

void TestStablePartitionCopyStencilDeviceSeq()
{
  TestStablePartitionCopyStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestStablePartitionCopyStencilDeviceSeq);

void TestStablePartitionCopyStencilDeviceDevice()
{
  TestStablePartitionCopyStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestStablePartitionCopyStencilDeviceDevice);

void TestStablePartitionCopyStencilDeviceNoSync()
{
  TestStablePartitionCopyStencilDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestStablePartitionCopyStencilDeviceNoSync);

void TestPartitionIfWithMagnitude(int magnitude)
{
  using offset_t = std::size_t;

  // Prepare input
  offset_t num_items = offset_t{1ull} << magnitude;
  thrust::counting_iterator<offset_t> begin(offset_t{0});
  auto end = begin + num_items;
  thrust::counting_iterator<offset_t> stencil(offset_t{0});
  ASSERT_EQUAL(static_cast<offset_t>(::cuda::std::distance(begin, end)), num_items);

  // Run algorithm on large number of items
  offset_t match_every_nth      = 1000000;
  offset_t expected_num_written = (num_items + match_every_nth - 1) / match_every_nth;

  // Tests input is correctly dereferenced for large offsets and selected items are correctly written
  {
    // Initialize input
    thrust::device_vector<offset_t> partitioned_out(expected_num_written);

    // Run test
    constexpr bool negate_matches = false;
    auto select_op                = mod_n<offset_t>{match_every_nth, negate_matches};
    auto partitioned_out_ends =
      thrust::stable_partition_copy(begin, end, partitioned_out.begin(), thrust::make_discard_iterator(), select_op);
    const auto selected_out_end = partitioned_out_ends.first;

    // Ensure number of selected items are correct
    const offset_t num_selected_out =
      static_cast<offset_t>(::cuda::std::distance(partitioned_out.begin(), selected_out_end));
    ASSERT_EQUAL(num_selected_out, expected_num_written);
    partitioned_out.resize(expected_num_written);

    // Ensure selected items are correct
    auto expected_out_it     = thrust::make_transform_iterator(begin, multiply_n<offset_t>{match_every_nth});
    bool all_results_correct = thrust::equal(partitioned_out.begin(), partitioned_out.end(), expected_out_it);
    ASSERT_EQUAL(all_results_correct, true);
  }

  // Tests input is correctly dereferenced for large offsets and rejected items are correctly written
  {
    // Initialize input
    thrust::device_vector<offset_t> partitioned_out(expected_num_written);

    // Run test
    constexpr bool negate_matches = true;
    auto select_op                = mod_n<offset_t>{match_every_nth, negate_matches};
    const auto partitioned_out_ends =
      thrust::stable_partition_copy(begin, end, thrust::make_discard_iterator(), partitioned_out.begin(), select_op);
    const auto rejected_out_end = partitioned_out_ends.second;

    // Ensure number of rejected items are correct
    const offset_t num_rejected_out =
      static_cast<offset_t>(::cuda::std::distance(partitioned_out.begin(), rejected_out_end));
    ASSERT_EQUAL(num_rejected_out, expected_num_written);
    partitioned_out.resize(expected_num_written);

    // Ensure rejected items are correct
    auto expected_out_it     = thrust::make_transform_iterator(begin, multiply_n<offset_t>{match_every_nth});
    bool all_results_correct = thrust::equal(partitioned_out.begin(), partitioned_out.end(), expected_out_it);
    ASSERT_EQUAL(all_results_correct, true);
  }
}

void TestPartitionIfWithLargeNumberOfItems()
{
  TestPartitionIfWithMagnitude(30);
  // These require 64-bit dispatches even when magnitude < 32.
#  ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestPartitionIfWithMagnitude(31);
  TestPartitionIfWithMagnitude(32);
  TestPartitionIfWithMagnitude(33);
#  endif
}
DECLARE_UNITTEST(TestPartitionIfWithLargeNumberOfItems);
#endif

template <typename ExecutionPolicy>
void TestPartitionCudaStreams(ExecutionPolicy policy)
{
  using Vector   = thrust::device_vector<int>;
  using T        = Vector::value_type;
  using Iterator = Vector::iterator;

  Vector data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 1;
  data[3] = 1;
  data[4] = 2;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  Iterator iter = thrust::partition(streampolicy, data.begin(), data.end(), is_even<T>());

  Vector ref(5);
  ref[0] = 2;
  ref[1] = 2;
  ref[2] = 1;
  ref[3] = 1;
  ref[4] = 1;

  ASSERT_EQUAL(iter - data.begin(), 2);
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}

void TestPartitionCudaStreamsSync()
{
  TestPartitionCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestPartitionCudaStreamsSync);

void TestPartitionCudaStreamsNoSync()
{
  TestPartitionCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestPartitionCudaStreamsNoSync);
