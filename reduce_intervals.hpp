#pragma once

#include <tbb/parallel_for.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/minmax.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/reduce.h>
#include <iostream>

namespace reduce_intervals_detail
{


template<typename L, typename R>
  inline L divide_ri(const L x, const R y)
{
  return (x + (y - 1)) / y;
}


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Size, typename BinaryFunction>
  struct body
{
  RandomAccessIterator1 first;
  RandomAccessIterator2 result;
  Size n, interval_size;
  BinaryFunction binary_op;

  body(RandomAccessIterator1 first, RandomAccessIterator2 result, Size n, Size interval_size, BinaryFunction binary_op)
    : first(first), result(result), n(n), interval_size(interval_size), binary_op(binary_op)
  {}

  void operator()(const tbb::blocked_range<Size> &r) const
  {
    Size interval_idx = r.begin();

    Size offset_to_first = interval_size * interval_idx;
    Size offset_to_last = thrust::min(n, offset_to_first + interval_size);

    RandomAccessIterator1 my_first = first + offset_to_first;
    RandomAccessIterator1 my_last  = first + offset_to_last;

    thrust::cpp::tag seq;

    // carefully pass the init value for the interval with raw_reference_cast
    typedef typename BinaryFunction::result_type sum_type;
    result[interval_idx] =
      thrust::reduce(seq, my_first + 1, my_last, sum_type(thrust::raw_reference_cast(*my_first)), binary_op);
  }
};


template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Size, typename BinaryFunction>
  body<RandomAccessIterator1,RandomAccessIterator2,Size,BinaryFunction>
    make_body(RandomAccessIterator1 first, RandomAccessIterator2 result, Size n, Size interval_size, BinaryFunction binary_op)
{
  return body<RandomAccessIterator1,RandomAccessIterator2,Size,BinaryFunction>(first, result, n, interval_size, binary_op);
}


}


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2, typename BinaryFunction>
  void reduce_intervals(RandomAccessIterator1 first,
                        RandomAccessIterator1 last,
                        Size interval_size,
                        RandomAccessIterator2 result,
                        BinaryFunction binary_op)
{
  typename thrust::iterator_difference<RandomAccessIterator1>::type n = last - first;

  Size num_intervals = reduce_intervals_detail::divide_ri(n, interval_size);

  tbb::parallel_for(::tbb::blocked_range<Size>(0, num_intervals, 1), reduce_intervals_detail::make_body(first, result, Size(n), interval_size, binary_op));
}


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
  void reduce_intervals(RandomAccessIterator1 first,
                        RandomAccessIterator1 last,
                        Size interval_size,
                        RandomAccessIterator2 result)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;

  return reduce_intervals(first, last, interval_size, result, thrust::plus<value_type>());
}

