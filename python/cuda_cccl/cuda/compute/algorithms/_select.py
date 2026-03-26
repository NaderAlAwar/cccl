# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .. import _bindings, types
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import call_build, set_cccl_iterator_state
from .._utils.protocols import get_data_pointer, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator


class _Select:
    __slots__ = [
        "build_result",
        "d_in_cccl",
        "d_flags_cccl",
        "d_out_cccl",
        "d_num_selected_out_cccl",
        "cond_cccl",
    ]

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        d_num_selected_out: DeviceArrayLike,
        cond: OpAdapter,
        d_flags: DeviceArrayLike | IteratorT | None = None,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_flags_cccl = (
            cccl.to_cccl_input_iter(d_flags) if d_flags is not None else None
        )
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)
        self.d_num_selected_out_cccl = cccl.to_cccl_output_iter(d_num_selected_out)

        value_type = cccl.get_value_type(d_flags if d_flags is not None else d_in)
        self.cond_cccl = cond.compile((value_type,), types.uint8)

        self.build_result = call_build(
            _bindings.DeviceSelectBuildResult,
            self.d_in_cccl,
            self.d_flags_cccl,
            self.d_out_cccl,
            self.d_num_selected_out_cccl,
            self.cond_cccl,
        )

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items: int,
        d_flags=None,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        if self.d_flags_cccl is not None:
            set_cccl_iterator_state(self.d_flags_cccl, d_flags)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        set_cccl_iterator_state(self.d_num_selected_out_cccl, d_num_selected_out)

        cond_adapter = make_op_adapter(cond)
        self.cond_cccl.state = cond_adapter.get_state()

        stream_handle = validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        return self.build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_flags_cccl,
            self.d_out_cccl,
            self.d_num_selected_out_cccl,
            self.cond_cccl,
            num_items,
            stream_handle,
        )


@cache_with_registered_key_functions
def make_select(
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
):
    """Creates a device-wide selection object using the unary predicate ``cond``.

    Example:
        Below, ``make_select`` is used to create a select object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array or iterator that will store the selected items
        d_num_selected_out: Device array that will store the number of selected items
        cond: Unary predicate used to decide whether each input item is selected.
            The signature is ``(T) -> bool``, where ``T`` is the data type of
            the input sequence.

    Returns:
        A callable object that can be used to perform the selection
    """
    cond_adapter = make_op_adapter(cond)
    return _Select(d_in, d_out, d_num_selected_out, cond_adapter)


@cache_with_registered_key_functions
def make_select_flagged(
    d_in: DeviceArrayLike | IteratorT,
    d_flags: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
):
    """Creates a device-wide selection object driven by a flags sequence.

    Example:
        Below, ``make_select_flagged`` is used to create a flagged select object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_flagged_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_flags: Device array or iterator containing the flags that control selection
        d_out: Device array or iterator that will store the selected items
        d_num_selected_out: Device array that will store the number of selected items
        cond: Unary predicate used to decide whether each flag selects its corresponding
            input item. The signature is ``(F) -> bool``, where ``F`` is the data
            type of the flags sequence.

    Returns:
        A callable object that can be used to perform flagged selection
    """
    cond_adapter = make_op_adapter(cond)
    return _Select(d_in, d_out, d_num_selected_out, cond_adapter, d_flags=d_flags)


def select(
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
    num_items: int,
    stream=None,
):
    """Performs device-wide selection using the single-phase API.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``select`` is used to select even numbers from an input array.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array or iterator that will store the selected items
        d_num_selected_out: Device array that will store the number of selected items
        cond: Unary predicate used to decide whether each input item is selected.
            The signature is ``(T) -> bool``, where ``T`` is the data type of
            the input sequence.
        num_items: Number of items to process
        stream: CUDA stream for the operation (optional)
    """
    selector = make_select(d_in, d_out, d_num_selected_out, cond)
    tmp_storage_bytes = selector(
        None, d_in, d_out, d_num_selected_out, cond, num_items, None, stream
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    selector(
        tmp_storage, d_in, d_out, d_num_selected_out, cond, num_items, None, stream
    )


def select_flagged(
    d_in: DeviceArrayLike | IteratorT,
    d_flags: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
    num_items: int,
    stream=None,
):
    """Performs device-wide flagged selection using the single-phase API.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``select_flagged`` is used to select values whose corresponding flags are nonzero.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_flagged_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_flags: Device array or iterator containing the flags that control selection
        d_out: Device array or iterator that will store the selected items
        d_num_selected_out: Device array that will store the number of selected items
        cond: Unary predicate used to decide whether each flag selects its corresponding
            input item. The signature is ``(F) -> bool``, where ``F`` is the data
            type of the flags sequence.
        num_items: Number of items to process
        stream: CUDA stream for the operation (optional)
    """
    selector = make_select_flagged(
        d_in,
        d_flags,
        d_out,
        d_num_selected_out,
        cond,
    )
    tmp_storage_bytes = selector(
        None,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items,
        d_flags,
        stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    selector(
        tmp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items,
        d_flags,
        stream,
    )
