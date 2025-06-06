//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// T shall be an object type other than cv in_place_t or cv nullopt_t
//   and shall satisfy the Cpp17Destructible requirements.
// Note: array types do not satisfy the Cpp17Destructible requirements.

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NonDestructible
{
  ~NonDestructible() = delete;
};

int main(int, char**)
{
  {
    cuda::std::optional<char&> o1; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                   // failed{{.*}}instantiation of optional with a reference type is ill-formed}}
    cuda::std::optional<NonDestructible> o2; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                             // failed{{.*}}instantiation of optional with a non-destructible type is
                                             // ill-formed}}
    cuda::std::optional<char[20]> o3; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                      // failed{{.*}}instantiation of optional with an array type is ill-formed}}
  }

  {
    cuda::std::optional<cuda::std::in_place_t> o1; // expected-error-re@optional:* {{{{(static_assert|static
                                                   // assertion)}} failed{{.*}}instantiation of optional with in_place_t
                                                   // is ill-formed}}
    cuda::std::optional<const cuda::std::in_place_t> o2; // expected-error-re@optional:* {{{{(static_assert|static
                                                         // assertion)}} failed{{.*}}instantiation of optional with
                                                         // in_place_t is ill-formed}}
    cuda::std::optional<volatile cuda::std::in_place_t> o3; // expected-error-re@optional:* {{{{(static_assert|static
                                                            // assertion)}} failed{{.*}}instantiation of optional with
                                                            // in_place_t is ill-formed}}
    cuda::std::optional<const volatile cuda::std::in_place_t> o4; // expected-error-re@optional:*
                                                                  // {{{{(static_assert|static assertion)}}
                                                                  // failed{{.*}}instantiation of optional with
                                                                  // in_place_t is ill-formed}}
  }

  {
    cuda::std::optional<cuda::std::nullopt_t> o1; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                                  // failed{{.*}}instantiation of optional with nullopt_t is
                                                  // ill-formed}}
    cuda::std::optional<const cuda::std::nullopt_t> o2; // expected-error-re@optional:* {{{{(static_assert|static
                                                        // assertion)}} failed{{.*}}instantiation of optional with
                                                        // nullopt_t is ill-formed}}
    cuda::std::optional<volatile cuda::std::nullopt_t> o3; // expected-error-re@optional:* {{{{(static_assert|static
                                                           // assertion)}} failed{{.*}}instantiation of optional with
                                                           // nullopt_t is ill-formed}}
    cuda::std::optional<const volatile cuda::std::nullopt_t> o4; // expected-error-re@optional:*
                                                                 // {{{{(static_assert|static assertion)}}
                                                                 // failed{{.*}}instantiation of optional with nullopt_t
                                                                 // is ill-formed}}
  }

  return 0;
}
