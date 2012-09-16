#include <utility>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>
#include <cassert>


template<typename L, typename R>
  inline L divide_ri(const L x, const R y)
{
  return (x + (y - 1)) / y;
}


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
  struct count_unique_body
{
  RandomAccessIterator1 first;
  RandomAccessIterator2 result;
  size_t n;
  size_t interval_size;

  count_unique_body(RandomAccessIterator1 first, RandomAccessIterator2 result, size_t n, size_t interval_size)
    : first(first), result(result), n(n), interval_size(interval_size)
  {}

  void operator()(const tbb::blocked_range<size_t> &r) const
  {
    assert(r.size() == 1);

    const size_t interval_idx = r.begin();

    size_t offset_to_first = interval_size * interval_idx;
    size_t offset_to_last  = thrust::min(n, offset_to_first + interval_size);

    RandomAccessIterator1 my_first = first + offset_to_first;
    RandomAccessIterator1 my_last  = first + offset_to_last;

    thrust::cpp::tag seq;

    result[interval_idx] =
      thrust::unique_copy(seq, my_first, my_last, thrust::make_discard_iterator()) -
      thrust::make_discard_iterator(0);
  }
};


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
  count_unique_body<RandomAccessIterator1,RandomAccessIterator2>
    make_body(RandomAccessIterator1 first, RandomAccessIterator2 result, size_t n, size_t interval_size)
{
  return count_unique_body<RandomAccessIterator1,RandomAccessIterator2>(first, result, n, interval_size);
}


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
  void count_unique_per_interval(RandomAccessIterator1 first,
                                 RandomAccessIterator1 last,
                                 Size interval_size,
                                 RandomAccessIterator2 result)
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = last - first;

  size_t num_intervals = divide_ri(n, interval_size);

  //tbb::parallel_for(::tbb::blocked_range<size_t>(0, last - first, interval_size), make_body(first, result, interval_size));
  tbb::parallel_for(::tbb::blocked_range<size_t>(0, num_intervals, 1), make_body(first, result, n, interval_size));
}


template<typename InputIterator1,
         typename InputIterator2>
  std::pair<
    InputIterator1,
    std::pair<
      typename InputIterator1::value_type,
      typename InputIterator2::value_type
    >
  >
    reduce_last_segment_backward(InputIterator1 keys_first,
                                 InputIterator1 keys_last,
                                 InputIterator2 values_first)
{
  size_t n = keys_last - keys_first;

  // reverse the ranges and consume from the end
  thrust::reverse_iterator<InputIterator1> keys_first_r(keys_last);
  thrust::reverse_iterator<InputIterator1> keys_last_r(keys_first);
  thrust::reverse_iterator<InputIterator2> values_first_r(values_first + n);

  typename InputIterator1::value_type result_key = *keys_first_r;
  typename InputIterator2::value_type result_value = *values_first_r;

  // consume the entirety of the first key's sequence
  for(++keys_first_r, ++values_first_r;
      (keys_first_r != keys_last_r) && (*keys_first_r == result_key);
      ++keys_first_r, ++values_first_r)
  {
    result_value = result_value + *values_first_r;
  }

  return std::make_pair(keys_first_r.base(), std::make_pair(result_key, result_value));
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  std::pair<
    typename OutputIterator1::value_type,
    typename OutputIterator2::value_type
  >
    reduce_by_key_with_carry(InputIterator1 keys_first, 
                             InputIterator1 keys_last,
                             InputIterator2 values_first,
                             OutputIterator1 keys_output,
                             OutputIterator2 values_output)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputKeyType;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type InputValueType;

  // first, consume the last sequence to produce the carry
  std::pair<
    typename OutputIterator1::value_type,
    typename OutputIterator2::value_type
  > result;

  std::tie(keys_last, result) = reduce_last_segment_backward(keys_first, keys_last, values_first);

  // finish with sequential reduce_by_key
  thrust::cpp::tag seq;
  thrust::reduce_by_key(seq, keys_first, keys_last, values_first, keys_output, values_output);
  
  return result;
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename Iterator7>
  struct serial_reduce_by_key_body
{
  Iterator1 keys_first;
  Iterator2 values_first;
  Iterator3 result_offset;
  Iterator4 keys_result;
  Iterator5 values_result;
  Iterator6 carry_result_key;
  Iterator7 carry_result_value;
  size_t n;
  size_t interval_size;

  serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result_key, Iterator7 carry_result_value, size_t n, size_t interval_size)
    : keys_first(keys_first), values_first(values_first), result_offset(result_offset), keys_result(keys_result), values_result(values_result), carry_result_key(carry_result_key), carry_result_value(carry_result_value), n(n), interval_size(interval_size)
  {}

  void operator()(const tbb::blocked_range<size_t> &r) const
  {
    assert(r.size() == 1);

    const size_t interval_idx = r.begin();

    const size_t offset_to_first = interval_size * interval_idx;
    const size_t offset_to_last = thrust::min(n, offset_to_first + interval_size);

    Iterator1 my_keys_first         = keys_first         + offset_to_first;
    Iterator1 my_keys_last          = keys_first         + offset_to_last;
    Iterator2 my_values_first       = values_first       + offset_to_first;
    Iterator3 my_result_offset      = result_offset      + interval_idx;
    Iterator4 my_keys_result        = keys_result        + *my_result_offset;
    Iterator5 my_values_result      = values_result      + *my_result_offset;
    Iterator6 my_carry_result_key   = carry_result_key   + interval_idx;
    Iterator7 my_carry_result_value = carry_result_value + interval_idx;

    std::cout << "interval " << interval_idx << " result offset: " << *my_result_offset << std::endl;

    // consume the rest of the interval with reduce_by_key
    thrust::cpp::tag seq;
    auto carry = reduce_by_key_with_carry(my_keys_first,
                                          my_keys_last,
                                          my_values_first,
                                          my_keys_result,
                                          my_values_result);

    *my_carry_result_key = carry.first;
    *my_carry_result_value = carry.second;
  }
};


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename Iterator7>
  serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,Iterator7>
    make_serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result_key, Iterator7 carry_result_value, size_t n, size_t interval_size)
{
  return serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,Iterator7>(keys_first, values_first, result_offset, keys_result, values_result, carry_result_key, carry_result_value, n, interval_size);
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
  std::pair<Iterator3,Iterator4>
    reduce_by_key(Iterator1 keys_first, Iterator1 keys_last, 
                  Iterator2 values_first,
                  Iterator3 keys_result,
                  Iterator4 values_result)
{
  // assume a quad core
  size_t p = 4;

  // generate O(P) intervals of sequential work
  typename thrust::iterator_difference<Iterator1>::type n = keys_last - keys_first;
  size_t interval_size = n / p;
  size_t num_intervals = divide_ri(n, interval_size);

  // decompose the input into intervals of size N / num_intervals
  std::vector<size_t> num_unique_keys_per_interval(num_intervals, 0);

  std::cout << "reduce_by_key(): interval_size: " << interval_size << std::endl;
  std::cout << "reduce_by_key(): num_intervals: " << num_intervals << std::endl;

  // count the number of unique keys in each interval
  count_unique_per_interval(keys_first, keys_last, interval_size, num_unique_keys_per_interval.begin());

  std::cout << "reduce_by_key(): num_unique_keys_per_interval: ";
  std::copy(num_unique_keys_per_interval.begin(), num_unique_keys_per_interval.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // scan the counts to get each body's offset
  // XXX this step is wrong
  // XXX we need to find the number of unique keys occurring before interval i
  // XXX NOT the sum of unique keys per interval before interval i
  std::vector<size_t> scatter_indices(num_intervals, 0);
  thrust::cpp::tag seq;
  thrust::exclusive_scan(seq,
                         num_unique_keys_per_interval.begin(), num_unique_keys_per_interval.end(), 
                         scatter_indices.begin(),
                         size_t(0));

  // do a reduce_by_key serially in each thread
  std::vector<typename Iterator1::value_type> carry_key(num_intervals);
  std::vector<typename Iterator2::value_type> carry_value(num_intervals, 0);
  tbb::parallel_for(::tbb::blocked_range<size_t>(0, num_intervals, 1),
    make_serial_reduce_by_key_body(keys_first, values_first, scatter_indices.begin(), keys_result, values_result, carry_key.begin(), carry_value.begin(), n, interval_size));

  std::cout << "reduce_by_key(): carry keys: ";
  std::copy(carry_key.begin(), carry_key.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "reduce_by_key(): carry values: ";
  std::copy(carry_value.begin(), carry_value.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // scatter the carry keys and accumulate the carry values
  for(int i = 0; i < scatter_indices.size(); ++i)
  {
    keys_result[num_unique_keys_per_interval[i] + scatter_indices[i] - 1] = carry_key[i];
    values_result[num_unique_keys_per_interval[i] + scatter_indices[i] - 1] += carry_value[i];
  }

  size_t size_of_result = scatter_indices.back() + num_unique_keys_per_interval.back();
  return std::make_pair(keys_result + size_of_result, values_result + size_of_result);
}

int main()
{
  size_t segment_size = 2;
  size_t n = 5 * segment_size;
  std::vector<int> keys(n);
  for(int i = 0; i < keys.size(); ++i)
  {
    keys[i] = i / segment_size;
  }

  std::vector<int> values(n, 1);

  std::cout << "main(): keys: ";
  std::copy(keys.begin(), keys.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "main(): values: ";
  std::copy(values.begin(), values.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::vector<int> keys_result(n / segment_size);
  std::vector<int> values_result(n / segment_size);

  auto ends = reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result.begin(), values_result.begin());

  keys_result.erase(ends.first, keys_result.end());
  values_result.erase(ends.second, values_result.end());

  std::cout << "main(): keys_result: ";
  std::copy(keys_result.begin(), keys_result.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "main(): values_result: ";
  std::copy(values_result.begin(), values_result.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::vector<int> keys_result_reference(n / segment_size);
  std::vector<int> values_result_reference(n / segment_size);
  auto ends1 = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result_reference.begin(), values_result_reference.begin());
  keys_result_reference.erase(ends1.first, keys_result_reference.end());
  values_result_reference.erase(ends1.second, values_result_reference.end());

  std::cout << "main(): keys_result_reference: ";
  std::copy(keys_result_reference.begin(), keys_result_reference.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "main(): values_result_reference: ";
  std::copy(values_result_reference.begin(), values_result_reference.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  assert(keys_result_reference == keys_result);
  assert(values_result_reference == values_result);

  return 0;
}

