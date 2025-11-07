#!/usr/bin/env python3
"""S5 Associative Scan Benchmark using cuda.compute with Custom Iterator."""

import ctypes
import sys
from functools import lru_cache
from typing import Tuple

import cuda.bench as bench
import cuda.compute
import numba
import numba.cuda
import numpy as np
import torch
from cuda.compute.iterators import _iterators
from numba import types
from numba.core import cgutils
from numba.core.extending import models, register_model
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import registry as cuda_lower_registry

# ============================================================================
# Custom DualRowPointerIterator for 2D S5 Scan
# ============================================================================


@lru_cache
def make_dual_row_pointer_iterator_struct():
    """Create ctypes struct for dual row pointer iterator state."""

    class DualRowPointerState(ctypes.Structure):
        _fields_ = [
            ("row_index", ctypes.c_int64),
            ("A_base_ptr", ctypes.c_void_p),
            ("Bu_base_ptr", ctypes.c_void_p),
            ("row_stride_bytes", ctypes.c_int64),
            ("element_size_bytes", ctypes.c_int64),
        ]

    return DualRowPointerState


def dual_row_pointer_iterator_numba_type(dtype):
    """Define numba type for dual row pointer iterator state."""

    class DualRowPointerViewType(types.Type):
        def __init__(self):
            super().__init__(name=f"DualRowPointerView_{dtype}")

    row_pointer_type = DualRowPointerViewType()

    int64_t = numba.from_dtype(np.int64)

    # Map numpy dtype to numba type
    dtype_map = {
        np.float16: types.float16,
        np.float32: types.float32,
        np.float64: types.float64,
    }
    element_type = dtype_map[dtype]
    element_ptr_t = types.CPointer(element_type)

    members = [
        ("row_index", int64_t),
        ("A_base_ptr", element_ptr_t),
        ("Bu_base_ptr", element_ptr_t),
        ("row_stride_bytes", int64_t),
        ("element_size_bytes", int64_t),
    ]

    class RowPointerAttrsTemplate(AttributeTemplate):
        pass

    def make_attr_resolver(ty):
        def resolve_fn(self, pp):
            return ty

        return resolve_fn

    for name, typ in members:
        setattr(RowPointerAttrsTemplate, f"resolve_{name}", make_attr_resolver(typ))

    @cuda_registry.register_attr
    class RowPointerAttrs(RowPointerAttrsTemplate):
        key = row_pointer_type

    ptr_type = types.CPointer(row_pointer_type)

    @cuda_registry.register_attr
    class PtrAttrs(AttributeTemplate):
        key = ptr_type

        def resolve_row_index(self, pp):
            return int64_t

    @register_model(DualRowPointerViewType)
    class DualRowPointerViewModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            super().__init__(dmm, fe_type, members)

    @cuda_lower_registry.lower_getattr_generic(row_pointer_type)
    def row_pointer_getattr(context, builder, sig, arg, attr):
        struct_values = cgutils.create_struct_proxy(row_pointer_type)(
            context, builder, value=arg
        )
        attr_ptr = struct_values._get_ptr_by_name(attr)
        attr_val = builder.load(attr_ptr)
        return attr_val

    @cuda_lower_registry.lower_setattr(ptr_type, "row_index")
    def row_pointer_set_row_index(context, builder, sig, args):
        data = builder.load(args[0])
        values = cgutils.create_struct_proxy(row_pointer_type)(
            context, builder, value=data
        )
        setattr(values, "row_index", args[1])
        return builder.store(values._getvalue(), args[0])

    @cuda_lower_registry.lower_getattr(ptr_type, "row_index")
    def row_pointer_get_row_index(context, builder, sig, arg):
        data = builder.load(arg)
        values = cgutils.create_struct_proxy(row_pointer_type)(
            context, builder, value=data
        )
        attr_ptr = values._get_ptr_by_name("row_index")
        attr_val = builder.load(attr_ptr)
        return attr_val

    return row_pointer_type


class DualRowPointerIteratorKind(_iterators.IteratorKind):
    pass


class DualRowPointerIterator(_iterators.IteratorBase):
    """
    Custom iterator for 2D S5 scan.

    Returns value tuples by computing row pointers and loading element-by-element.
    """

    iterator_kind_type = DualRowPointerIteratorKind

    def __init__(
        self,
        A_base_ptr: int,
        Bu_base_ptr: int,
        row_stride_bytes: int,
        element_size_bytes: int,
        dtype,
        state_dim: int,
    ):
        state_numba_type = dual_row_pointer_iterator_numba_type(dtype)

        # Map numpy dtype to numba type for value_type
        dtype_map = {
            np.float16: types.float16,
            np.float32: types.float32,
            np.float64: types.float64,
        }
        element_type = dtype_map[dtype]
        value_type = types.UniTuple(types.UniTuple(element_type, state_dim), 2)

        DualRowPointerState_cls = make_dual_row_pointer_iterator_struct()
        host_state = DualRowPointerState_cls(
            0, A_base_ptr, Bu_base_ptr, row_stride_bytes, element_size_bytes
        )

        super().__init__(
            cvalue=host_state,
            state_type=state_numba_type,
            value_type=value_type,
        )
        self.state_dim = state_dim

    @staticmethod
    def advance(state_ref, distance):
        """Advance by distance rows."""
        state_ref.row_index = state_ref.row_index + distance

    @property
    def input_dereference(self):
        return self.input_dereference_impl

    @property
    def output_dereference(self):
        return self.output_dereference_impl

    @staticmethod
    def input_dereference_impl(state_ref, result):
        """Load row values element-by-element."""
        zero = numba.int32(0)
        state = state_ref[zero]
        row_idx = state.row_index

        elements_per_row = state.row_stride_bytes // state.element_size_bytes
        A_row_ptr = state.A_base_ptr + row_idx * elements_per_row
        Bu_row_ptr = state.Bu_base_ptr + row_idx * elements_per_row

        # Hardcoded for state_dim - would need code generation for arbitrary sizes
        result[0] = (
            (A_row_ptr[0], A_row_ptr[1], A_row_ptr[2], A_row_ptr[3]),
            (Bu_row_ptr[0], Bu_row_ptr[1], Bu_row_ptr[2], Bu_row_ptr[3]),
        )

    @staticmethod
    def output_dereference_impl(state_ref, value_tuples):
        """Write value tuples element-by-element."""
        zero = numba.int32(0)
        state = state_ref[zero]
        row_idx = state.row_index

        elements_per_row = state.row_stride_bytes // state.element_size_bytes
        A_row_ptr = state.A_base_ptr + row_idx * elements_per_row
        Bu_row_ptr = state.Bu_base_ptr + row_idx * elements_per_row

        A_vals = value_tuples[0]
        Bu_vals = value_tuples[1]

        # Hardcoded writes
        A_row_ptr[0] = A_vals[0]
        A_row_ptr[1] = A_vals[1]
        A_row_ptr[2] = A_vals[2]
        A_row_ptr[3] = A_vals[3]

        Bu_row_ptr[0] = Bu_vals[0]
        Bu_row_ptr[1] = Bu_vals[1]
        Bu_row_ptr[2] = Bu_vals[2]
        Bu_row_ptr[3] = Bu_vals[3]


# ============================================================================
# S5 Scan Implementation
# ============================================================================


def s5_op(x_vals, y_vals):
    """S5 operator on value tuples."""
    x_A, x_Bu = x_vals[0], x_vals[1]
    y_A, y_Bu = y_vals[0], y_vals[1]

    # Hardcoded for state_dim=4 - unrolled for Numba compatibility
    return (
        (
            y_A[0] * x_A[0],
            y_A[1] * x_A[1],
            y_A[2] * x_A[2],
            y_A[3] * x_A[3],
        ),
        (
            y_A[0] * x_Bu[0] + y_Bu[0],
            y_A[1] * x_Bu[1] + y_Bu[1],
            y_A[2] * x_Bu[2] + y_Bu[2],
            y_A[3] * x_Bu[3] + y_Bu[3],
        ),
    )


def create_s5_scan_operator(dtype, state_dim):
    """Create a reusable S5 scan operator for the given dtype."""
    # Create dummy arrays to set up iterators with the right types
    dummy_A = np.zeros((1, state_dim), dtype=dtype)
    dummy_Bu = np.zeros((1, state_dim), dtype=dtype)

    d_dummy_A = numba.cuda.to_device(dummy_A)
    d_dummy_Bu = numba.cuda.to_device(dummy_Bu)

    A_ptr = d_dummy_A.device_ctypes_pointer.value
    Bu_ptr = d_dummy_Bu.device_ctypes_pointer.value
    row_stride_bytes = state_dim * dummy_A.dtype.itemsize
    element_size_bytes = dummy_A.dtype.itemsize

    input_iter = DualRowPointerIterator(
        A_ptr,
        Bu_ptr,
        row_stride_bytes,
        element_size_bytes,
        dummy_A.dtype.type,
        state_dim,
    )
    output_iter = DualRowPointerIterator(
        A_ptr,
        Bu_ptr,
        row_stride_bytes,
        element_size_bytes,
        dummy_A.dtype.type,
        state_dim,
    )

    # Create and cache the scan operator
    scan_op = cuda.compute.make_inclusive_scan(
        d_in=input_iter,
        d_out=output_iter,
        op=s5_op,
        init_value=None,
    )

    return scan_op, input_iter, output_iter


def s5_scan_cuda_compute(
    d_A_in: np.ndarray, d_Bu_in: np.ndarray, scan_op, stream=None
) -> Tuple[numba.cuda.devicearray.DeviceNDArray, numba.cuda.devicearray.DeviceNDArray]:
    """2D S5 scan using custom iterator - single scan call."""
    timesteps, state_dim = d_A_in.shape

    # Convert bench.CudaStream to numba stream for numba.cuda operations
    numba_stream = None
    if stream is not None:
        # Get the stream pointer and create a numba external stream
        stream_ptr = stream.addressof()
        numba_stream = numba.cuda.external_stream(stream_ptr)

    d_A_out = numba.cuda.device_array(
        (timesteps, state_dim), dtype=d_A_in.dtype, stream=numba_stream
    )
    d_Bu_out = numba.cuda.device_array(
        (timesteps, state_dim), dtype=d_Bu_in.dtype, stream=numba_stream
    )

    A_in_ptr = d_A_in.device_ctypes_pointer.value
    Bu_in_ptr = d_Bu_in.device_ctypes_pointer.value
    A_out_ptr = d_A_out.device_ctypes_pointer.value
    Bu_out_ptr = d_Bu_out.device_ctypes_pointer.value
    row_stride_bytes = state_dim * d_A_in.dtype.itemsize
    element_size_bytes = d_A_in.dtype.itemsize

    input_iter = DualRowPointerIterator(
        A_in_ptr,
        Bu_in_ptr,
        row_stride_bytes,
        element_size_bytes,
        d_A_in.dtype.type,
        state_dim,
    )
    output_iter = DualRowPointerIterator(
        A_out_ptr,
        Bu_out_ptr,
        row_stride_bytes,
        element_size_bytes,
        d_A_in.dtype.type,
        state_dim,
    )

    # Allocate temporary storage
    temp_storage_bytes = scan_op(
        None,
        input_iter,
        output_iter,
        num_items=timesteps,
        init_value=None,
        stream=stream,
    )
    d_temp_storage = torch.empty(temp_storage_bytes, dtype=torch.uint8, device="cuda")

    # Execute the scan
    scan_op(
        d_temp_storage,
        input_iter,
        output_iter,
        num_items=timesteps,
        init_value=None,
        stream=stream,
    )

    return d_A_out, d_Bu_out


def as_torch_cuda_Stream(
    cs: bench.CudaStream, dev: int | None
) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


# ============================================================================
# Benchmark Functions
# ============================================================================


def s5_associative_scan(state: bench.State):
    """Benchmark cuda.compute implementation."""
    dtype_str = state.get_string("dtype")
    timesteps = state.get_int64("Timesteps")
    state_dim = 4  # Currently hardcoded in DualRowPointerIterator

    dtype_map = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
    dtype = dtype_map[dtype_str]

    state.add_summary("dtype", dtype_str)
    state.add_summary("state_dim", state_dim)

    A_in = np.random.randn(timesteps, state_dim).astype(dtype)
    Bu_in = np.random.randn(timesteps, state_dim).astype(dtype)

    # Create the scan operator once (will be cached)
    scan_op, _, _ = create_s5_scan_operator(dtype, state_dim)

    d_A_in = numba.cuda.to_device(A_in)
    d_Bu_in = numba.cuda.to_device(Bu_in)

    # Warmup
    s5_scan_cuda_compute(d_A_in, d_Bu_in, scan_op)

    def launcher(launch: bench.Launch):
        s5_scan_cuda_compute(d_A_in, d_Bu_in, scan_op, stream=launch.get_stream())

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    # Register benchmarks
    b = bench.register(s5_associative_scan)

    # Pointer types are now parameterized by dtype
    # b.add_string_axis("dtype", ["float16", "float32", "float64"])
    b.add_string_axis("dtype", ["float32", "float64"])
    b.add_int64_power_of_two_axis("Timesteps", range(12, 29, 4))

    bench.run_all_benchmarks(sys.argv)
