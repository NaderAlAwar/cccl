#pragma once

#include <iostream>
#include "system.h"

namespace unittest
{

template<typename T>
  const char *type_name(void)
{
  return demangle(typeid(T).name());
} // end type_name()

} // end unittest

template <typename Iterator>
void PRINT(Iterator first, Iterator last)
{
  size_t n = 0;
  for (Iterator i = first; i != last; i++, n++)
    std::cout << ">>> [" << n << "] = " << *i << std::endl;
}

template <typename Container>
void PRINT(const Container& c)
{
  PRINT(c.begin(), c.end());
}

template <size_t N>
void PRINT(const char (&c)[N])
{
  std::cout << std::string(c, c + N) << std::endl;
}

