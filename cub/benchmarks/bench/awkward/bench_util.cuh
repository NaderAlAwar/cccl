#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <random>

template <typename T>
static void print_array(const thrust::device_vector<T>& d_values)
{
  thrust::host_vector<T> h_values = d_values;

  for (T x : h_values)
  {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

template <typename T>
static void print_array(const thrust::device_vector<T>& d_values, const thrust::device_vector<int>& d_offsets)
{
  thrust::host_vector<T> h_values    = d_values;
  thrust::host_vector<int> h_offsets = d_offsets;

  int num_segments = static_cast<int>(h_offsets.size()) - 1;

  for (int seg = 0; seg < num_segments; ++seg)
  {
    int start = h_offsets[seg];
    int end   = h_offsets[seg + 1];

    std::cout << "Segment " << seg << ": ";
    for (int i = start; i < end; ++i)
    {
      std::cout << h_values[i] << " ";
    }
    std::cout << std::endl;
  }
}

thrust::host_vector<int> partition_into_segments(size_t num_elements, int max_segment_size)
{
  thrust::host_vector<int> sizes;

  std::random_device rd;
  std::mt19937 gen(rd());

  int remaining = num_elements;
  while (remaining > 0)
  {
    int max_size = std::min(max_segment_size, remaining);
    std::uniform_int_distribution<int> dist(1, max_size);
    int size = dist(gen);

    sizes.push_back(size);
    remaining -= size;
  }

  return sizes;
}
