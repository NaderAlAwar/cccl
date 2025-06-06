#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/partition.h>

#include <unittest/unittest.h>

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x) const
  {
    return ((int) x % 2) == 0;
  }
};

template <typename Vector>
void TestPartitionPointSimple()
{
  using Iterator = typename Vector::iterator;

  Vector v{1, 1, 1, 0};

  Iterator first = v.begin();

  Iterator last = v.begin() + 4;
  Iterator ref  = first + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(first, last, ::cuda::std::identity{}));

  last = v.begin() + 3;
  ref  = last;
  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(first, last, ::cuda::std::identity{}));
}
DECLARE_VECTOR_UNITTEST(TestPartitionPointSimple);

template <class Vector>
void TestPartitionPoint()
{
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  Iterator ref = thrust::stable_partition(v.begin(), v.end(), is_even<T>());

  ASSERT_EQUAL(ref - v.begin(), thrust::partition_point(v.begin(), v.end(), is_even<T>()) - v.begin());
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPartitionPoint);

template <typename ForwardIterator, typename Predicate>
ForwardIterator partition_point(my_system& system, ForwardIterator first, ForwardIterator, Predicate)
{
  system.validate_dispatch();
  return first;
}

void TestPartitionPointDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::partition_point(sys, vec.begin(), vec.begin(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestPartitionPointDispatchExplicit);

template <typename ForwardIterator, typename Predicate>
ForwardIterator partition_point(my_tag, ForwardIterator first, ForwardIterator, Predicate)
{
  *first = 13;
  return first;
}

void TestPartitionPointDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::partition_point(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestPartitionPointDispatchImplicit);

struct test_less_than
{
  long long expected;

  _CCCL_DEVICE bool operator()(long long y)
  {
    return y < expected;
  }
};

void TestPartitionPointWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(0);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  test_less_than fn = {(1ll << magnitude) - 17};

  ASSERT_EQUAL(::cuda::std::distance(begin, thrust::partition_point(thrust::device, begin, end, fn)),
               (1ll << magnitude) - 17);
}

void TestPartitionPointWithBigIndexes()
{
  TestPartitionPointWithBigIndexesHelper(30);
  TestPartitionPointWithBigIndexesHelper(31);
  TestPartitionPointWithBigIndexesHelper(32);
  TestPartitionPointWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestPartitionPointWithBigIndexes);
