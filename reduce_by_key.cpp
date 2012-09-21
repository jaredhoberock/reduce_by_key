#include <thrust/random.h>
#include <vector>
#include <iostream>
#include <cassert>
#include "reduce_by_key.hpp"


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

  auto ends = experimental::reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_result.begin(), values_result.begin(), thrust::equal_to<int>(), thrust::plus<int>());
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

