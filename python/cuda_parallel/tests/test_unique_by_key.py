# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import cupy as cp
import numpy as np
import numba.cuda
import pytest

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators
from cuda.parallel.experimental.struct import gpu_struct


DTYPE_LIST = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        return rng.random(size=size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def type_to_problem_sizes(dtype) -> List[int]:
    if dtype in [np.uint8, np.int8]:
        return [2, 4, 5, 6]
    elif dtype in [np.uint16, np.int16]:
        return [4, 8, 14]
    elif dtype in [np.uint32, np.int32, np.float32]:
        return [4, 10, 20]
    elif dtype in [np.uint64, np.int64, np.float64]:
        return [4, 10, 20]
    else:
        raise ValueError("Unsupported dtype")


def unique_by_key_device(
    d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, op, num_items, stream=None
):
    unique_by_key = algorithms.unique_by_key(
        d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, op
    )

    temp_storage_size = unique_by_key(
        None, d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, num_items, stream=stream
    )
    d_temp_storage = numba.cuda.device_array(
        temp_storage_size, dtype=np.uint8, stream=stream.ptr if stream else 0
    )
    unique_by_key(
        d_temp_storage,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        num_items,
        stream=stream,
    )


def is_equal_func(lhs, rhs):
    return lhs == rhs


def unique_by_key_host(
    keys, items, is_equal=is_equal_func
):
    # Must implement our own version of unique_by_key since np.unique() returns
    # unique elements across the entire array, while cub::UniqueByKey
    # de-duplicates consecutive keys that are equal.
    if len(keys) == 0:
        return np.empty(0), np.empty(0)

    prev_key = keys[0]
    keys_out = [prev_key]
    items_out = [items[0]]

    for idx, (previous, next) in enumerate(zip(keys, keys[1:])):
        if not is_equal(previous, next):
            keys_out.append(next)

            # add 1 since we are enumerating over pairs
            items_out.append(items[idx + 1])

    return np.array(keys_out), np.array(items_out)


def compare_op(lhs, rhs):
    return np.uint8(lhs == rhs)


@pytest.mark.parametrize(
    "dtype",
    DTYPE_LIST,
)
def test_unique_by_key(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2
        num_items = 10

        # h_in_keys = random_array(num_items, dtype, max_value=20)
        h_in_keys = np.array([4, 3, 3, 1, 2, 6, 3, 3, 9, 7])
        h_in_items = random_array(num_items, np.float32)
        h_out_keys = np.empty(num_items, dtype=dtype)
        h_out_items = np.empty(num_items, dtype=np.float32)
        h_out_num_selected = np.empty(1, np.int32)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_items = numba.cuda.to_device(h_in_items)
        d_out_keys = numba.cuda.to_device(h_out_keys)
        d_out_items = numba.cuda.to_device(h_out_items)
        d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

        unique_by_key_device(d_in_keys, d_in_items, d_out_keys,
                             d_out_items, d_out_num_selected, compare_op, num_items)

        h_out_num_selected = d_out_num_selected.copy_to_host()
        num_selected = h_out_num_selected[0]
        h_out_keys = np.resize(d_out_keys.copy_to_host(), num_selected)
        h_out_items = np.resize(d_out_items.copy_to_host(), num_selected)

        expected_out_keys, expected_out_items = unique_by_key_host(
            h_in_keys, h_in_items)

        np.testing.assert_array_equal(h_out_keys, expected_out_keys)
        np.testing.assert_array_equal(h_out_items, expected_out_items)


@pytest.mark.parametrize(
    "dtype",
    DTYPE_LIST,
)
def test_unique_by_key_iterators(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype, max_value=20)
        h_in_items = random_array(num_items, np.float32)
        h_out_keys = np.empty(num_items, dtype=dtype)
        h_out_items = np.empty(num_items, dtype=np.float32)
        h_out_num_selected = np.empty(1, np.int32)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_items = numba.cuda.to_device(h_in_items)
        d_out_keys = numba.cuda.to_device(h_out_keys)
        d_out_items = numba.cuda.to_device(h_out_items)
        d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

        i_in_keys = iterators.CacheModifiedInputIterator(
            d_in_keys, modifier="stream", prefix="keys"
        )
        i_in_items = iterators.CacheModifiedInputIterator(
            d_in_items, modifier="stream", prefix="items"
        )

        unique_by_key_device(i_in_keys, i_in_items, d_out_keys,
                             d_out_items, d_out_num_selected, compare_op, num_items)

        h_out_num_selected = d_out_num_selected.copy_to_host()
        num_selected = h_out_num_selected[0]
        h_out_keys = np.resize(d_out_keys.copy_to_host(), num_selected)
        h_out_items = np.resize(d_out_items.copy_to_host(), num_selected)

        expected_out_keys, expected_out_items = unique_by_key_host(
            h_in_keys, h_in_items)

        np.testing.assert_array_equal(h_out_keys, expected_out_keys)
        np.testing.assert_array_equal(h_out_items, expected_out_items)


def test_unique_by_key_complex():
    def compare_complex(lhs, rhs):
        return np.uint8(lhs.real == rhs.real)

    num_items = 100000
    max_value = 20
    real = random_array(num_items, np.int64, max_value)
    imaginary = random_array(num_items, np.int64, max_value)

    h_in_keys = real + 1j * imaginary
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=h_in_keys.dtype)
    h_out_items = np.empty(num_items, dtype=np.float32)
    h_out_num_selected = np.empty(1, np.int32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_out_keys = numba.cuda.to_device(h_out_keys)
    d_out_items = numba.cuda.to_device(h_out_items)
    d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

    unique_by_key_device(d_in_keys, d_in_items, d_out_keys,
                         d_out_items, d_out_num_selected, compare_complex, num_items)

    h_out_num_selected = d_out_num_selected.copy_to_host()
    num_selected = h_out_num_selected[0]
    h_out_keys = np.resize(d_out_keys.copy_to_host(), num_selected)
    h_out_items = np.resize(d_out_items.copy_to_host(), num_selected)

    expected_out_keys, expected_out_items = unique_by_key_host(
        h_in_keys, h_in_items, compare_complex)

    np.testing.assert_array_equal(h_out_keys, expected_out_keys)
    np.testing.assert_array_equal(h_out_items, expected_out_items)


@pytest.mark.xfail(
    reason="Creating an array of gpu_struct keys does not work currently (see https://github.com/NVIDIA/cccl/issues/3789)"
)
def test_unique_by_key_struct_types():
    @gpu_struct
    class key_pair:
        a: np.int16
        b: np.uint64

    @gpu_struct
    class item_pair:
        a: np.int32
        b: np.float32

    def struct_compare_op(lhs, rhs):
        return np.uint8((lhs.a == rhs.a) and (lhs.b == rhs.b))

    num_items = 10000

    a_keys = np.random.randint(0, 20, num_items, dtype=np.int16)
    b_keys = np.random.randint(0, 20, num_items, dtype=np.uint64)

    a_items = np.random.randint(0, 20, num_items, dtype=np.int32)
    b_items = np.random.rand(num_items).astype(np.float32)

    h_in_keys = np.empty(num_items, dtype=key_pair.dtype)
    h_in_items = np.empty(num_items, dtype=item_pair.dtype)
    h_out_num_selected = np.empty(1, np.int32)

    h_in_keys["a"] = a_keys
    h_in_keys["b"] = b_keys

    h_in_items["a"] = a_items
    h_in_items["b"] = b_items

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_keys = cp.asarray(d_in_keys).view(key_pair.dtype)
    d_in_items = numba.cuda.to_device(h_in_items)
    d_in_items = cp.asarray(d_in_items).view(item_pair.dtype)

    d_out_keys = cp.empty_like(d_in_keys)
    d_out_items = cp.empty_like(d_in_items)
    d_out_num_selected = numba.cuda.to_device(h_out_num_selected)

    unique_by_key_device(d_in_keys, d_in_items, d_out_keys,
                         d_out_items, d_out_num_selected, struct_compare_op, num_items)

    h_out_num_selected = d_out_num_selected.copy_to_host()
    num_selected = h_out_num_selected[0]
    h_out_keys = np.resize(d_out_keys.copy_to_host(), num_selected)
    h_out_items = np.resize(d_out_items.copy_to_host(), num_selected)

    expected_out_keys, expected_out_items = unique_by_key_host(
        h_in_keys, h_in_items, struct_compare_op)

    np.testing.assert_array_equal(h_out_keys, expected_out_keys)
    np.testing.assert_array_equal(h_out_items, expected_out_items)


def test_unique_by_key_with_stream(cuda_stream):
    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    num_items = 10000

    h_in_keys = random_array(num_items, np.int32, max_value=20)
    h_in_items = random_array(num_items, np.float32)
    h_out_keys = np.empty(num_items, dtype=np.int32)
    h_out_items = np.empty(num_items, dtype=np.float32)
    h_out_num_selected = np.empty(1, np.int32)

    with cp_stream:
        h_in_keys = random_array(num_items, np.int32)
        d_in_keys = cp.asarray(h_in_keys)
        d_in_items = cp.asarray(h_in_items)
        d_out_keys = cp.empty_like(h_out_keys)
        d_out_items = cp.empty_like(h_out_items)
        d_out_num_selected = cp.empty_like(h_out_num_selected)

    unique_by_key_device(d_in_keys, d_in_items, d_out_keys,
                         d_out_items, d_out_num_selected, compare_op, num_items, stream=cuda_stream)

    h_out_keys = d_out_keys.get()
    h_out_items = d_out_items.get()
    h_out_num_selected = d_out_num_selected.get()

    num_selected = h_out_num_selected[0]
    h_out_keys = np.resize(h_out_keys, num_selected)
    h_out_items = np.resize(h_out_items, num_selected)

    expected_out_keys, expected_out_items = unique_by_key_host(
        h_in_keys, h_in_items)

    np.testing.assert_array_equal(h_out_keys, expected_out_keys)
    np.testing.assert_array_equal(h_out_items, expected_out_items)
