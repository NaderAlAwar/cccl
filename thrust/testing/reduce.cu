#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>

#include <limits>

#include <unittest/unittest.h>

template <typename T>
struct plus_mod_10
{
  _CCCL_HOST_DEVICE T operator()(T lhs, T rhs) const
  {
    return ((lhs % 10) + (rhs % 10)) % 10;
  }
};

template <class Vector>
void TestReduceSimple()
{
  using T = typename Vector::value_type;

  Vector v{1, -2, 3};

  // no initializer
  ASSERT_EQUAL(thrust::reduce(v.begin(), v.end()), 2);

  // with initializer
  ASSERT_EQUAL(thrust::reduce(v.begin(), v.end(), (T) 10), 12);
}
DECLARE_VECTOR_UNITTEST(TestReduceSimple);

template <typename InputIterator>
int reduce(my_system& system, InputIterator, InputIterator)
{
  system.validate_dispatch();
  return 13;
}

void TestReduceDispatchExplicit()
{
  thrust::device_vector<int> vec;

  my_system sys(0);
  thrust::reduce(sys, vec.begin(), vec.end());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReduceDispatchExplicit);

template <typename InputIterator>
int reduce(my_tag, InputIterator, InputIterator)
{
  return 13;
}

void TestReduceDispatchImplicit()
{
  thrust::device_vector<int> vec;

  int result = thrust::reduce(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQUAL(13, result);
}
DECLARE_UNITTEST(TestReduceDispatchImplicit);

template <typename T>
struct TestReduce
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
    T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestReduce, IntegralTypes> TestReduceInstance;

template <class IntVector, class FloatVector>
void TestReduceMixedTypes()
{
  // make sure we get types for default args and operators correct
  IntVector int_input{1, 2, 3, 4};

  FloatVector float_input{1.5, 2.5, 3.5, 4.5};

  // float -> int should use using plus<int> operator by default
  ASSERT_EQUAL(thrust::reduce(float_input.begin(), float_input.end(), (int) 0), 10);

  // int -> float should use using plus<float> operator by default
  ASSERT_EQUAL(thrust::reduce(int_input.begin(), int_input.end(), (float) 0.5), 10.5);
}
void TestReduceMixedTypesHost()
{
  TestReduceMixedTypes<thrust::host_vector<int>, thrust::host_vector<float>>();
}
DECLARE_UNITTEST(TestReduceMixedTypesHost);
void TestReduceMixedTypesDevice()
{
  TestReduceMixedTypes<thrust::device_vector<int>, thrust::device_vector<float>>();
}
DECLARE_UNITTEST(TestReduceMixedTypesDevice);

template <typename T>
struct TestReduceWithOperator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 3;

    T cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init, plus_mod_10<T>());
    T gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init, plus_mod_10<T>());

    ASSERT_EQUAL(cpu_result, gpu_result);
  }
};
VariableUnitTest<TestReduceWithOperator, UnsignedIntegralTypes> TestReduceWithOperatorInstance;

template <typename T>
struct plus_mod3
{
  T* table;

  plus_mod3(T* table)
      : table(table)
  {}

  _CCCL_HOST_DEVICE T operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

template <typename Vector>
void TestReduceWithIndirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};

  Vector table{0, 1, 2, 0, 1, 2};

  T result = thrust::reduce(data.begin(), data.end(), T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQUAL(result, T(1));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestReduceWithIndirection);

template <typename T>
void TestReduceCountingIterator()
{
  size_t const n = 15 * sizeof(T);

  ASSERT_LEQUAL(T(n), unittest::truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);

  T init = unittest::random_integer<T>();

  T h_result = thrust::reduce(h_first, h_first + n, init);
  T d_result = thrust::reduce(d_first, d_first + n, init);

  // we use ASSERT_ALMOST_EQUAL because we're testing floating point types
  ASSERT_ALMOST_EQUAL(h_result, d_result);
}
DECLARE_GENERIC_UNITTEST(TestReduceCountingIterator);

void TestReduceWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  long long result = thrust::reduce(thrust::device, begin, end);

  ASSERT_EQUAL(result, 1ll << magnitude);
}

void TestReduceWithBigIndexes()
{
  TestReduceWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestReduceWithBigIndexesHelper(31);
  TestReduceWithBigIndexesHelper(32);
  TestReduceWithBigIndexesHelper(33);
#endif
}
DECLARE_UNITTEST(TestReduceWithBigIndexes);
