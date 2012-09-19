#include <thrust/iterator/reverse_iterator.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <vector>
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


template<typename InputIterator1,
         typename InputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::pair<
    InputIterator1,
    thrust::pair<
      typename InputIterator1::value_type,
      typename InputIterator2::value_type
    >
  >
    reduce_last_segment_backward(InputIterator1 keys_first,
                                 InputIterator1 keys_last,
                                 InputIterator2 values_first,
                                 BinaryPredicate binary_pred,
                                 BinaryFunction binary_op)
{
  typename thrust::iterator_difference<InputIterator1>::type n = keys_last - keys_first;

  // reverse the ranges and consume from the end
  thrust::reverse_iterator<InputIterator1> keys_first_r(keys_last);
  thrust::reverse_iterator<InputIterator1> keys_last_r(keys_first);
  thrust::reverse_iterator<InputIterator2> values_first_r(values_first + n);

  typename InputIterator1::value_type result_key = *keys_first_r;
  typename InputIterator2::value_type result_value = *values_first_r;

  // consume the entirety of the first key's sequence
  for(++keys_first_r, ++values_first_r;
      (keys_first_r != keys_last_r) && binary_pred(*keys_first_r, result_key);
      ++keys_first_r, ++values_first_r)
  {
    result_value = binary_op(result_value, *values_first_r);
  }

  return thrust::make_pair(keys_first_r.base(), std::make_pair(result_key, result_value));
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::tuple<
    OutputIterator1,
    OutputIterator2,
    typename OutputIterator1::value_type,
    typename OutputIterator2::value_type
  >
    reduce_by_key_with_carry(InputIterator1 keys_first, 
                             InputIterator1 keys_last,
                             InputIterator2 values_first,
                             OutputIterator1 keys_output,
                             OutputIterator2 values_output,
                             BinaryPredicate binary_pred,
                             BinaryFunction binary_op)
{
  // first, consume the last sequence to produce the carry
  // XXX is there an elegant way to pose this such that we don't need to default construct carry?
  thrust::pair<
    typename OutputIterator1::value_type,
    typename OutputIterator2::value_type
  > carry;

  thrust::tie(keys_last, carry) = reduce_last_segment_backward(keys_first, keys_last, values_first, binary_pred, binary_op);

  // finish with sequential reduce_by_key
  thrust::cpp::tag seq;
  thrust::tie(keys_output, values_output) =
    thrust::reduce_by_key(seq, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
  
  return thrust::make_tuple(keys_output, values_output, carry.first, carry.second);
}


template<typename Iterator>
  bool interval_has_carry(size_t interval_idx, size_t interval_size, size_t num_intervals, Iterator tail_flags)
{
  // to discover whether the interval has a carry, look at the tail_flag corresponding to its last element 
  // the final interval never has a carry by definition
  return (interval_idx + 1 < num_intervals) ? !tail_flags[(interval_idx + 1) * interval_size - 1] : false;
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename BinaryPredicate, typename BinaryFunction>
  struct serial_reduce_by_key_body
{
  typedef typename thrust::iterator_difference<Iterator1>::type size_type;

  Iterator1 keys_first;
  Iterator2 values_first;
  Iterator3 result_offset;
  Iterator4 keys_result;
  Iterator5 values_result;
  Iterator6 carry_result;

  size_type n;
  size_type interval_size;
  size_type num_intervals;

  BinaryPredicate binary_pred;
  BinaryFunction binary_op;

  serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result, size_type n, size_type interval_size, size_type num_intervals, BinaryPredicate binary_pred, BinaryFunction binary_op)
    : keys_first(keys_first), values_first(values_first),
      result_offset(result_offset),
      keys_result(keys_result),
      values_result(values_result),
      carry_result(carry_result),
      n(n),
      interval_size(interval_size),
      num_intervals(num_intervals),
      binary_pred(binary_pred),
      binary_op(binary_op)
  {}

  void operator()(const ::tbb::blocked_range<size_type> &r) const
  {
    assert(r.size() == 1);

    const size_type interval_idx = r.begin();

    const size_type offset_to_first = interval_size * interval_idx;
    const size_type offset_to_last = thrust::min(n, offset_to_first + interval_size);

    Iterator1 my_keys_first     = keys_first    + offset_to_first;
    Iterator1 my_keys_last      = keys_first    + offset_to_last;
    Iterator2 my_values_first   = values_first  + offset_to_first;
    Iterator3 my_result_offset  = result_offset + interval_idx;
    Iterator4 my_keys_result    = keys_result   + *my_result_offset;
    Iterator5 my_values_result  = values_result + *my_result_offset;
    Iterator6 my_carry_result   = carry_result  + interval_idx;

    // consume the rest of the interval with reduce_by_key
    typedef typename thrust::iterator_value<Iterator1>::type key_type;
    typedef typename thrust::iterator_value<Iterator2>::type value_type;

    // XXX is there a way to pose this so that we don't require default construction of carry?
    thrust::pair<key_type, value_type> carry;

    thrust::tie(my_keys_result, my_values_result, carry.first, carry.second) =
      reduce_by_key_with_carry(my_keys_first,
                               my_keys_last,
                               my_values_first,
                               my_keys_result,
                               my_values_result,
                               binary_pred,
                               binary_op);

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


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename BinaryPredicate, typename BinaryFunction>
  serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,BinaryPredicate,BinaryFunction>
    make_serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 result_offset, Iterator4 keys_result, Iterator5 values_result, Iterator6 carry_result, typename thrust::iterator_difference<Iterator1>::type n, size_t interval_size, size_t num_intervals, BinaryPredicate binary_pred, BinaryFunction binary_op)
{
  return serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,BinaryPredicate,BinaryFunction>(keys_first, values_first, result_offset, keys_result, values_result, carry_result, n, interval_size, num_intervals, binary_pred, binary_op);
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename BinaryFunction>
  thrust::pair<Iterator3,Iterator4>
    reduce_by_key(Iterator1 keys_first, Iterator1 keys_last, 
                  Iterator2 values_first,
                  Iterator3 keys_result,
                  Iterator4 values_result,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
{

  typedef typename thrust::iterator_difference<Iterator1>::type difference_type;
  difference_type n = keys_last - keys_first;
  if(n == 0) return std::make_pair(keys_result, values_result);

  // XXX this value is a tuning opportunity
  const difference_type parallelism_threshold = 10000;

  if(n < parallelism_threshold)
  {
    // don't bother parallelizing for small n
    thrust::cpp::tag seq;
    return thrust::reduce_by_key(seq, keys_first, keys_last, values_first, keys_result, values_result, binary_pred, binary_op);
  }

  // count the number of processors
  const unsigned int p = std::max(1u, ::tbb::tbb_thread::hardware_concurrency());

  // generate O(P) intervals of sequential work
  // XXX oversubscribing is a tuning opportunity
  const unsigned int subscription_rate = 1;
  difference_type interval_size = std::min(parallelism_threshold, std::max<difference_type>(n, n / (subscription_rate * p)));
  difference_type num_intervals = divide_ri(n, interval_size);

  //std::clog << "reduce_by_key(): interval_size: " << interval_size << std::endl;
  //std::clog << "reduce_by_key(): num_intervals: " << num_intervals << std::endl;

  // decompose the input into intervals of size N / num_intervals
  // add one extra element to this vector to store the size of the entire result
  std::vector<difference_type> interval_output_offsets(num_intervals + 1, 0);

  // first count the number of tail flags in each interval
  auto tail_flags = make_tail_flags(keys_first, keys_last, binary_pred);
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
  // the final interval never has a carry by definition, so don't reserve space for it
  std::vector<typename Iterator2::value_type> carries(num_intervals - 1, 0);

  // force grainsize == 1 with simple_partioner()
  ::tbb::parallel_for(::tbb::blocked_range<difference_type>(0, num_intervals, 1),
    make_serial_reduce_by_key_body(keys_first, values_first, interval_output_offsets.begin(), keys_result, values_result, carries.begin(), n, interval_size, num_intervals, binary_pred, binary_op),
    ::tbb::simple_partitioner());

  //std::clog << "reduce_by_key(): carries: ";
  //std::copy(carries.begin(), carries.end(), std::ostream_iterator<int>(std::clog, " "));
  //std::clog << std::endl;

  difference_type size_of_result = interval_output_offsets.back();

  //std::clog << "reduce_by_key(): partial values: ";
  //std::copy(values_result, values_result + size_of_result, std::ostream_iterator<int>(std::clog, " "));
  //std::clog << std::endl;

  // sequentially accumulate the carries
  // note that the last interval does not have a carry
  for(int i = 0; i < carries.size(); ++i)
  {
    // if our interval has a carry, then we need to sum the carry to the next interval's output offset
    // if it does not have a carry, then we need to ignore carry_value[i]
    if(interval_has_carry(i, interval_size, num_intervals, tail_flags.begin()))
    {
      int output_idx = interval_output_offsets[i+1];

      values_result[output_idx] = binary_op(values_result[output_idx], carries[i]);
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


void test(size_t n)
{
  std::vector<int> keys(n);
  std::vector<int> values(n, 1);

  std::cout << "generating test case for n = " << n << "... " << std::flush;
  generate_test_case(keys.begin(), keys.end(), values.begin());
  std::cout << "done." << std::endl;

  // silence for large n
  if(n < 100)
  {
    std::clog.rdbuf((n < 100) ? std::clog.rdbuf() : 0);

    std::clog << "test(): keys      : ";
    std::copy(keys.begin(), keys.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "test(): tail_flags: ";
    auto flags = make_tail_flags(keys.begin(), keys.end());
    std::copy(flags.begin(), flags.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "test(): values: ";
    std::copy(values.begin(), values.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;
  }

  std::vector<int> keys_result(n);
  std::vector<int> values_result(n, 13);

  auto ends = ::reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result.begin(), values_result.begin(), thrust::equal_to<int>(), thrust::plus<int>());
  keys_result.erase(ends.first, keys_result.end());
  values_result.erase(ends.second, values_result.end());

  if(n < 100)
  {
    std::clog << "test(): keys_result: ";
    std::copy(keys_result.begin(), keys_result.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "test(): values_result: ";
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
    std::clog << "test(): keys_result_reference: ";
    std::copy(keys_result_reference.begin(), keys_result_reference.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;

    std::clog << "test(): values_result_reference: ";
    std::copy(values_result_reference.begin(), values_result_reference.end(), std::ostream_iterator<int>(std::clog, " "));
    std::clog << std::endl;
  }

  assert(keys_result_reference == keys_result);
  assert(values_result_reference == values_result);

  std::cout << "test passed." << std::endl;
}


int main()
{
  std::vector<size_t> standard_test_sizes = 
          {0, 1, 2, 3, 4, 5, 8, 10, 13, 16, 17, 19, 27, 30, 31, 32,
           33, 35, 42, 53, 58, 63, 64, 65, 72, 97, 100, 127, 128, 129, 142, 183, 192, 201, 240, 255, 256,
           257, 302, 511, 512, 513, 687, 900, 1023, 1024, 1025, 1565, 1786, 1973, 2047, 2048, 2049, 3050, 4095, 4096,
           4097, 5030, 7791, 10000, 10027, 12345, 16384, 17354, 26255, 32768, 43718, 65533, 65536,
           65539, 123456, 131072, 731588, 1048575, 1048576,
           3398570, 9760840, (1 << 24) - 1, (1 << 24),
           (1 << 24) + 1, (1 << 25) - 1, (1 << 25), (1 << 25) + 1, (1 << 26) - 1, 1 << 26,
           (1 << 26) + 1, (1 << 27) - 1, (1 << 27)};

  for(int i = 0; i < standard_test_sizes.size(); ++i)
  {
    test(standard_test_sizes[i]);
  }

  return 0;
}

