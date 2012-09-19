#include <utility>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <vector>
#include <iterator>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb_thread.h>
#include <iostream>
#include <cassert>
#include "head_flags.hpp"
#include "tail_flags.hpp"
#include "reduce_intervals.hpp"


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
  std::tuple<
    OutputIterator1,
    OutputIterator2,
    typename OutputIterator1::value_type,
    typename OutputIterator2::value_type
  >
    reduce_by_key_with_carry(InputIterator1 keys_first, 
                             InputIterator1 keys_last,
                             InputIterator2 values_first,
                             OutputIterator1 keys_output,
                             OutputIterator2 values_output)
{
  // first, consume the last sequence to produce the carry
  std::pair<
    typename OutputIterator1::value_type,
    typename OutputIterator2::value_type
  > result;

  std::tie(keys_last, result) = reduce_last_segment_backward(keys_first, keys_last, values_first);

  // finish with sequential reduce_by_key
  thrust::cpp::tag seq;
  thrust::tie(keys_output, values_output) = thrust::reduce_by_key(seq, keys_first, keys_last, values_first, keys_output, values_output);
  
  return std::make_tuple(keys_output, values_output, result.first, result.second);
}


template<typename Iterator>
  bool interval_has_carry(size_t interval_idx, size_t interval_size, size_t num_intervals, Iterator tail_flags)
{
  // to discover whether the interval has a carry, look at the tail_flag corresponding to its last element 
  // the final interval never has a carry by definition
  return (interval_idx + 1 < num_intervals) ? !tail_flags[(interval_idx + 1) * interval_size - 1] : false;
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6>
  struct serial_reduce_by_key_body
{
  Iterator1 keys_first;
  Iterator2 values_first;
  Iterator3 result_offset;
  Iterator4 keys_result;
  Iterator5 values_result;
  Iterator6 carry_result;
  size_t n;
  size_t interval_size;
  size_t num_intervals;

  serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result, size_t n, size_t interval_size, size_t num_intervals)
    : keys_first(keys_first), values_first(values_first), result_offset(result_offset), keys_result(keys_result), values_result(values_result), carry_result(carry_result), n(n), interval_size(interval_size), num_intervals(num_intervals)
  {}

  void operator()(const tbb::blocked_range<size_t> &r) const
  {
    assert(r.size() == 1);

    const size_t interval_idx = r.begin();

    const size_t offset_to_first = interval_size * interval_idx;
    const size_t offset_to_last = thrust::min(n, offset_to_first + interval_size);

    Iterator1 my_keys_first     = keys_first    + offset_to_first;
    Iterator1 my_keys_last      = keys_first    + offset_to_last;
    Iterator2 my_values_first   = values_first  + offset_to_first;
    Iterator3 my_result_offset  = result_offset + interval_idx;
    Iterator4 my_keys_result    = keys_result   + *my_result_offset;
    Iterator5 my_values_result  = values_result + *my_result_offset;
    Iterator6 my_carry_result   = carry_result  + interval_idx;

    // consume the rest of the interval with reduce_by_key
    typedef typename std::iterator_traits<Iterator1>::value_type key_type;
    typedef typename std::iterator_traits<Iterator2>::value_type value_type;
    std::pair<key_type, value_type> carry;

    thrust::cpp::tag seq;
    auto iterators_and_carry = reduce_by_key_with_carry(my_keys_first,
                                                        my_keys_last,
                                                        my_values_first,
                                                        my_keys_result,
                                                        my_values_result);
    
    std::tie(my_keys_result, my_values_result, carry.first, carry.second) =
      reduce_by_key_with_carry(my_keys_first,
                               my_keys_last,
                               my_values_first,
                               my_keys_result,
                               my_values_result);

    // store to carry only when we actually have a carry
    // store to my_keys_result & my_values_result otherwise
    
    // create tail_flags so we can check for a carry
    auto tail_flags = make_tail_flags(keys_first, keys_first + n);

    if(interval_has_carry(interval_idx, interval_size, num_intervals, tail_flags.begin()))
    {
      // we can ignore the carry's key
      *my_carry_result = carry.second;
    }
    else
    {
      *my_keys_result = carry.first;
      *my_values_result = carry.second;
    }
  }
};


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6>
  serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6>
    make_serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result, size_t n, size_t interval_size, size_t num_intervals)
{
  return serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6>(keys_first, values_first, result_offset, keys_result, values_result, carry_result, n, interval_size, num_intervals);
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
  std::pair<Iterator3,Iterator4>
    reduce_by_key(Iterator1 keys_first, Iterator1 keys_last, 
                  Iterator2 values_first,
                  Iterator3 keys_result,
                  Iterator4 values_result)
{
  // count the number of processors
  const size_t p = tbb::tbb_thread::hardware_concurrency();

  // generate O(P) intervals of sequential work
  typename thrust::iterator_difference<Iterator1>::type n = keys_last - keys_first;
  size_t interval_size = n / p;
  size_t num_intervals = divide_ri(n, interval_size);

  //std::clog << "reduce_by_key(): interval_size: " << interval_size << std::endl;
  //std::clog << "reduce_by_key(): num_intervals: " << num_intervals << std::endl;

  // decompose the input into intervals of size N / num_intervals
  // add one extra element to this vector to store the size of the entire result
  std::vector<size_t> interval_output_offsets(num_intervals + 1, 0);

  // first count the number of tail flags in each interval
  auto tail_flags = make_tail_flags(keys_first, keys_last);
  reduce_intervals(tail_flags.begin(), tail_flags.end(), interval_size, interval_output_offsets.begin() + 1, thrust::plus<size_t>());

  //std::clog << "reduce_by_key(): num_tail_flags_per_interval: ";
  //std::copy(interval_output_offsets.begin(), interval_output_offsets.end(), std::ostream_iterator<int>(std::clog, " "));
  //std::clog << std::endl;

  // scan the counts to get each body's output offset
  thrust::cpp::tag seq;
  thrust::inclusive_scan(seq,
                         interval_output_offsets.begin() + 1, interval_output_offsets.end(), 
                         interval_output_offsets.begin() + 1);

  //std::clog << "reduce_by_key(): interval_output_offsets: ";
  //std::copy(interval_output_offsets.begin(), interval_output_offsets.end(), std::ostream_iterator<int>(std::clog, " "));
  //std::clog << std::endl;

  // do a reduce_by_key serially in each thread
  // the final interval never has a carry by definition
  std::vector<typename Iterator2::value_type> carry_value(num_intervals - 1, 0);
  tbb::parallel_for(::tbb::blocked_range<size_t>(0, num_intervals, 1),
    make_serial_reduce_by_key_body(keys_first, values_first, interval_output_offsets.begin(), keys_result, values_result, carry_value.begin(), n, interval_size, num_intervals));

  //std::clog << "reduce_by_key(): carry values: ";
  //std::copy(carry_value.begin(), carry_value.end(), std::ostream_iterator<int>(std::clog, " "));
  //std::clog << std::endl;

  size_t size_of_result = interval_output_offsets.back();

  //std::clog << "reduce_by_key(): partial values: ";
  //std::copy(values_result, values_result + size_of_result, std::ostream_iterator<int>(std::clog, " "));
  //std::clog << std::endl;

  // accumulate the carry values
  // note that the last interval does not have a carry
  for(int i = 0; i < carry_value.size(); ++i)
  {
    // if our interval has a carry, then we need to sum the carry to the next interval's output offset
    // if it does not have a carry, then we need to ignore carry_value[i]
    if(interval_has_carry(i, interval_size, num_intervals, tail_flags.begin()))
    {
      int output_idx = interval_output_offsets[i+1];

      values_result[output_idx] += carry_value[i];
    }
  }

  return std::make_pair(keys_result + size_of_result, values_result + size_of_result);
}


template<typename Iterator1, typename Iterator2>
  void generate_test_case(Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first)
{
  size_t n = keys_last - keys_first;

  size_t entropy = 3;

  thrust::default_random_engine rng(13);
  for(size_t i = 0, k = 0; i < n; ++i)
  {
    keys_first[i] = k;
    if(rng() % entropy == 0)
      k++;
  }

  for(size_t i = 0; i < n; ++i)
  {
    values_first[i] = rng() % entropy;
  }
}


int main()
{
  //size_t n = 13;
  size_t n = 100000000;
  std::vector<int> keys(n);
  std::vector<int> values(n, 1);

  std::cout << "generating test case... ";
  generate_test_case(keys.begin(), keys.end(), values.begin());
  std::cout << "done." << std::endl;

  // silence for large n
  if(n < 100)
  {
    std::clog.rdbuf((n < 100) ? std::clog.rdbuf() : 0);

    std::clog << "main(): keys      : ";
    std::copy(keys.begin(), keys.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "main(): tail_flags: ";
    auto flags = make_tail_flags(keys.begin(), keys.end());
    std::copy(flags.begin(), flags.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "main(): values: ";
    std::copy(values.begin(), values.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;
  }

  std::vector<int> keys_result(n);
  std::vector<int> values_result(n, 13);

  auto ends = reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result.begin(), values_result.begin());
  keys_result.erase(ends.first, keys_result.end());
  values_result.erase(ends.second, values_result.end());

  if(n < 100)
  {
    std::clog << "main(): keys_result: ";
    std::copy(keys_result.begin(), keys_result.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "main(): values_result: ";
    std::copy(values_result.begin(), values_result.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;
  }

  std::vector<int> keys_result_reference(n);
  std::vector<int> values_result_reference(n);
  auto ends1 = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result_reference.begin(), values_result_reference.begin());
  keys_result_reference.erase(ends1.first, keys_result_reference.end());
  values_result_reference.erase(ends1.second, values_result_reference.end());

  if(n < 100)
  {
    std::clog << "main(): keys_result_reference: ";
    std::copy(keys_result_reference.begin(), keys_result_reference.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "main(): values_result_reference: ";
    std::copy(values_result_reference.begin(), values_result_reference.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;
  }

  assert(keys_result_reference == keys_result);
  assert(values_result_reference == values_result);

  std::cout << "test passed." << std::endl;

  return 0;
}

