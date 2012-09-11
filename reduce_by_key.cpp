#include <utility>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


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

    size_t i = r.begin() / grainsize;

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

  tbb::parallel_for(::tbb::blocked_range<size_t>(0, interval_size), make_body(first, result, interval_size));
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

  if(keys_first != keys_last)
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

  typename InputIterator1::difference_type num_remaining = keys_last - keys_first;

  thrust::cpp::tag seq;
  return thrust::reduce(seq, values_first + 1, values_first + num_remaining, *values_first);
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
    size_t i = r.begin() / grainsize;

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

  // decompose the input into intervals of size N / P
  std::vector<size_t> num_unique_keys_per_interval(p, 0);

  typename thrust::iterator_difference<Iterator1>::type n = keys_last - keys_first;
  size_t interval_size = divide_ri(n,p);

  // count the number of unique keys in each interval
  count_unique_per_interval(keys_first, keys_last, interval_size, num_unique_keys_per_interval.begin());

  // scan the counts to get each body's offset
  std::vector<size_t> scatter_indices(p, 0);
  thrust::cpp::tag seq;
  thrust::exclusive_scan(num_unique_keys_per_interval.begin(), num_unique_keys_per_interval.end(), 
                         scatter_indices.begin(),
                         size_t(0));

  // do a reduce_by_key serially in each thread
  std::vector<typename Iterator2::value_type> carry(p);
  tbb::parallel_for(::tbb::blocked_range<size_t>(0, interval_size),
    make_serial_reduce_by_key_body(keys_first, values_first, num_unique_keys_per_interval.begin(), scatter_indices.begin(), keys_result, values_result, carry.begin(), interval_size));

  // accumulate the carries
  for(int i = 1; i < scatter_indices.size(); ++i)
  {
    values_result[scatter_indices[i]] += carry[i];
  }

  size_t size_of_result = scatter_indices.back() + num_unique_keys_per_interval.back();
  values_result[size_of_result-1] += carry.back();

  return std::make_pair(keys_result + size_of_result, values_result + size_of_result);
}

int main()
{
  std::vector<int> keys(8);
  for(int i = 0; i < keys.size(); ++i)
  {
    keys[i] = i / 2;
  }

  std::vector<int> values(8, 1);

  std::vector<int> keys_result(4);
  std::vector<int> values_result(4);

  reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result.begin(), values_result.begin());

  return 0;
}

