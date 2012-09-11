#include <utility>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <iostream>


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
  size_t grainsize;

  count_unique_body(RandomAccessIterator1 first, RandomAccessIterator2 result, size_t grainsize)
    : first(first), result(result), grainsize(grainsize)
  {}

  void operator()(const tbb::blocked_range<size_t> &r) const
  {
    thrust::cpp::tag seq;

    size_t i = divide_ri(r.begin(), grainsize);

    result[i] =
      thrust::unique_copy(seq, first + r.begin(), first + r.end(), thrust::make_discard_iterator()) -
      thrust::make_discard_iterator(0);
  }
};


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
  count_unique_body<RandomAccessIterator1,RandomAccessIterator2>
    make_body(RandomAccessIterator1 first, RandomAccessIterator2 result, size_t grainsize)
{
  return count_unique_body<RandomAccessIterator1,RandomAccessIterator2>(first, result, grainsize);
}


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
  void count_unique_per_interval(RandomAccessIterator1 first,
                                 RandomAccessIterator1 last,
                                 Size interval_size,
                                 RandomAccessIterator2 result)
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = last - first;
  tbb::parallel_for(::tbb::blocked_range<size_t>(0, last - first, interval_size), make_body(first, result, interval_size));
}


template<typename InputIterator1,
         typename Size,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  typename OutputIterator2::value_type
    reduce_by_key_with_carry(InputIterator1 keys_first, 
                             InputIterator1 keys_last,
                             Size num_unique_keys,
                             InputIterator2 values_first,
                             OutputIterator1 keys_output,
                             OutputIterator2 values_output)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type  InputKeyType;
  typedef typename thrust::iterator_traits<InputIterator2>::value_type  InputValueType;

  typedef typename InputIterator2::value_type TemporaryType;

  if(keys_first != keys_last && num_unique_keys > 1)
  {
    InputKeyType  temp_key   = *keys_first;
    TemporaryType temp_value = *values_first;

    for(++keys_first, ++values_first;
        (keys_first != keys_last) && (num_unique_keys > 1);
        ++keys_first, ++values_first, --num_unique_keys)
    {
      InputKeyType    key  = *keys_first;
      InputValueType value = *values_first;

      if (temp_key == key)
      {
        temp_value = temp_value + value;
      }
      else
      {
        *keys_output   = temp_key;
        *values_output = temp_value;

        ++keys_output;
        ++values_output;

        temp_key   = key;
        temp_value = value;
      }
    }

    *keys_output   = temp_key;
    *values_output = temp_value;

    ++keys_output;
    ++values_output;
  }


  typename OutputIterator2::value_type result = 0;

  typename InputIterator1::difference_type num_remaining = keys_last - keys_first;
  if(num_remaining)
  {
    //std::cout << "reduce_by_key_with_carry(): num_remaining: " << num_remaining << std::endl;

    thrust::cpp::tag seq;
    result = thrust::reduce(seq, values_first + 1, values_first + num_remaining, *values_first);
  }
  
  return result;
}


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename Iterator7>
  struct serial_reduce_by_key_body
{
  Iterator1 keys_first;
  Iterator2 values_first;
  Iterator3 num_unique_keys;
  Iterator4 result_offset;
  Iterator5 keys_result;
  Iterator6 values_result;
  Iterator7 carry_result;
  size_t grainsize;

  serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 num_unique_keys, Iterator4 result_offset, Iterator5 keys_result, Iterator6 values_result, Iterator7 carry_result, size_t grainsize)
    : keys_first(keys_first), values_first(values_first), num_unique_keys(num_unique_keys), result_offset(result_offset), keys_result(keys_result), values_result(values_result), carry_result(carry_result), grainsize(grainsize)
  {}

  void operator()(const tbb::blocked_range<size_t> &r) const
  {
    size_t i = divide_ri(r.begin(), grainsize);

    Iterator1 my_keys_first = keys_first + r.begin();
    Iterator1 my_keys_last  = keys_first + r.end();
    Iterator3 my_num_unique_keys = num_unique_keys + i;
    Iterator2 my_values_first = values_first + r.begin();
    Iterator4 my_result_offset = result_offset + i;
    Iterator5 my_keys_result = keys_result + *my_result_offset;
    Iterator6 my_values_result = values_result + *my_result_offset;
    Iterator7 my_carry_result = carry_result + i;

    // consume the rest of the interval with reduce_by_key
    thrust::cpp::tag seq;
    *my_carry_result = reduce_by_key_with_carry(my_keys_first,
                                                my_keys_last,
                                                *my_num_unique_keys,
                                                my_values_first,
                                                my_keys_result,
                                                my_values_result);
  }
};


template<typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6, typename Iterator7>
  serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,Iterator7>
    make_serial_reduce_by_key_body(Iterator1 keys_first, Iterator2 values_first, Iterator3 num_unique_keys, Iterator4 result_offset, Iterator5 keys_result, Iterator6 values_result, Iterator7 carry_result, size_t grainsize)
{
  return serial_reduce_by_key_body<Iterator1,Iterator2,Iterator3,Iterator4,Iterator5,Iterator6,Iterator7>(keys_first, values_first, num_unique_keys, result_offset, keys_result, values_result, carry_result, grainsize);
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
  std::vector<size_t> scatter_indices(num_intervals, 0);
  thrust::cpp::tag seq;
  thrust::exclusive_scan(seq,
                         num_unique_keys_per_interval.begin(), num_unique_keys_per_interval.end(), 
                         scatter_indices.begin(),
                         size_t(0));

  // do a reduce_by_key serially in each thread
  std::vector<typename Iterator2::value_type> carry(num_intervals, 0);
  tbb::parallel_for(::tbb::blocked_range<size_t>(0, keys_last - keys_first, interval_size),
    make_serial_reduce_by_key_body(keys_first, values_first, num_unique_keys_per_interval.begin(), scatter_indices.begin(), keys_result, values_result, carry.begin(), interval_size));

  std::cout << "reduce_by_key(): carries: ";
  std::copy(carry.begin(), carry.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // accumulate the carries
  for(int i = 0; i < scatter_indices.size(); ++i)
  {
    values_result[num_unique_keys_per_interval[i] + scatter_indices[i] - 1] += carry[i];
  }

  size_t size_of_result = scatter_indices.back() + num_unique_keys_per_interval.back();
  return std::make_pair(keys_result + size_of_result, values_result + size_of_result);
}

int main()
{
  size_t segment_size = 2;
  size_t n = 4 * segment_size;
  std::vector<int> keys(n);
  for(int i = 0; i < keys.size(); ++i)
  {
    keys[i] = i / segment_size;
  }

  std::vector<int> values(n, 1);

  std::vector<int> keys_result(n / segment_size);
  std::vector<int> values_result(n / segment_size);

  std::cout << "main(): keys: ";
  std::copy(keys.begin(), keys.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "main(): values: ";
  std::copy(values.begin(), values.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  auto ends = reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result.begin(), values_result.begin());

  keys_result.erase(ends.first, keys_result.end());
  values_result.erase(ends.second, values_result.end());

  std::cout << "main(): keys_result: ";
  std::copy(keys_result.begin(), keys_result.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "main(): values_result: ";
  std::copy(values_result.begin(), values_result.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}

