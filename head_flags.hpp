#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

template<typename RandomAccessIterator,
         typename BinaryPredicate = thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator>::type>,
         typename IndexType = typename thrust::iterator_difference<RandomAccessIterator>::type>
  class head_flags
{
  private:
    struct head_flag_functor
    {
      BinaryPredicate binary_pred;

      typedef bool result_type;

      head_flag_functor()
        : binary_pred()
      {}

      head_flag_functor(BinaryPredicate binary_pred)
        : binary_pred(binary_pred)
      {}

      template<typename Tuple>
      __host__ __device__ __thrust_forceinline__
      result_type operator()(const Tuple &t)
      {
        const IndexType i = thrust::get<0>(t);

        // note that we do not dereference the tuple's 2nd element when i == 0
        // and therefore do not dereference a bad location at the boundary
        return (i == 0 || !binary_pred(thrust::get<1>(t), thrust::get<2>(t)));
      }
    };

    typedef thrust::counting_iterator<IndexType> counting_iterator;

  public:
    typedef thrust::transform_iterator<
      head_flag_functor,
      thrust::zip_iterator<thrust::tuple<counting_iterator,RandomAccessIterator,RandomAccessIterator> >
    > iterator;

    head_flags(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor())),
        m_end(m_begin + (last - first))
    {}

    head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(binary_pred))),
        m_end(m_begin + (last - first))
    {}

    iterator begin() const
    {
      return m_begin;
    }

    iterator end() const
    {
      return m_end;
    }

  private:
    iterator m_begin, m_end;
};


template<typename RandomAccessIterator>
  head_flags<RandomAccessIterator>
    make_head_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return head_flags<RandomAccessIterator>(first, last);
}

