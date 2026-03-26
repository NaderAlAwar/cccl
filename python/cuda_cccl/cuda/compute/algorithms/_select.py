# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .. import _bindings, types
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import call_build, set_cccl_iterator_state
from .._utils import protocols
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator


def _get_temp_storage_state(temp_storage):
    if temp_storage is None:
        return 0, 0
    return temp_storage.nbytes, protocols.get_data_pointer(temp_storage)


class _SelectBase:
    __slots__ = [
        "build_result",
        "d_in_cccl",
        "d_flags_cccl",
        "d_out_cccl",
        "d_num_selected_out_cccl",
        "cond_cccl",
    ]

    build_result_type = None

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

        build_args = [self.d_in_cccl]
        if self.d_flags_cccl is not None:
            build_args.append(self.d_flags_cccl)
        build_args.extend(
            [self.d_out_cccl, self.d_num_selected_out_cccl, self.cond_cccl]
        )

        self.build_result = call_build(self.build_result_type, *build_args)

    def _call_impl(
        self,
        temp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items: int,
        stream=None,
        d_flags=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        if self.d_flags_cccl is not None:
            set_cccl_iterator_state(self.d_flags_cccl, d_flags)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        set_cccl_iterator_state(self.d_num_selected_out_cccl, d_num_selected_out)

        cond_adapter = make_op_adapter(cond)
        self.cond_cccl.state = cond_adapter.get_state()

        stream_handle = protocols.validate_and_get_stream(stream)
        temp_storage_bytes, d_temp_storage = _get_temp_storage_state(temp_storage)

        if self.d_flags_cccl is None:
            return self.build_result.compute(
                d_temp_storage,
                temp_storage_bytes,
                self.d_in_cccl,
                self.d_out_cccl,
                self.d_num_selected_out_cccl,
                self.cond_cccl,
                num_items,
                stream_handle,
            )

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


class _Select(_SelectBase):
    build_result_type = _bindings.DeviceSelectIfBuildResult

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        d_num_selected_out: DeviceArrayLike,
        cond: OpAdapter,
    ):
        super().__init__(d_in, d_out, d_num_selected_out, cond)

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items: int,
        stream=None,
    ):
        return self._call_impl(
            temp_storage,
            d_in,
            d_out,
            d_num_selected_out,
            cond,
            num_items,
            stream,
        )


class _SelectFlagged(_SelectBase):
    build_result_type = _bindings.DeviceSelectFlaggedIfBuildResult

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_flags: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        d_num_selected_out: DeviceArrayLike,
        cond: OpAdapter,
    ):
        super().__init__(d_in, d_out, d_num_selected_out, cond, d_flags=d_flags)

    def __call__(
        self,
        temp_storage,
        d_in,
        d_flags,
        d_out,
        d_num_selected_out,
        cond,
        num_items: int,
        stream=None,
    ):
        return self._call_impl(
            temp_storage,
            d_in,
            d_out,
            d_num_selected_out,
            cond,
            num_items,
            stream,
            d_flags=d_flags,
        )


def _run_select(selector, temp_storage_args):
    tmp_storage_bytes = selector(None, *temp_storage_args)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, temp_storage_args[-1])
    selector(tmp_storage, *temp_storage_args)


@cache_with_registered_key_functions
def make_select(
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
):
    """
    Create a select object that can be called to select elements matching a condition.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_object.py
            :language: python
            :start-after: # example-begin
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
    """
    Create a select_flagged object that can be called to select elements whose
    corresponding flags satisfy a condition.
    """
    cond_adapter = make_op_adapter(cond)
    return _SelectFlagged(d_in, d_flags, d_out, d_num_selected_out, cond_adapter)


def select(
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
    num_items: int,
    stream=None,
):
    """
    Select all input elements for which ``cond`` evaluates to true.
    """
    cond_adapter = make_op_adapter(cond)
    selector = make_select(d_in, d_out, d_num_selected_out, cond_adapter)
    _run_select(
        selector,
        (d_in, d_out, d_num_selected_out, cond_adapter, num_items, stream),
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
    """
    Select input elements whose corresponding flags satisfy ``cond``.
    """
    cond_adapter = make_op_adapter(cond)
    selector = make_select_flagged(
        d_in,
        d_flags,
        d_out,
        d_num_selected_out,
        cond_adapter,
    )
    _run_select(
        selector,
        (d_in, d_flags, d_out, d_num_selected_out, cond_adapter, num_items, stream),
    )
