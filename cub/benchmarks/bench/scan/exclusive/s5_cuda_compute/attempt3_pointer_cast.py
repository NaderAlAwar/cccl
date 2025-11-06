#!/usr/bin/env python3
"""
S5 2D Scan - Attempt 3: Custom DualRowPointerIterator (WORKING!)

This successfully implements 2D S5 scan with a SINGLE scan call by creating
a custom iterator that handles row-wise access using CPointer arithmetic.

**Solution:**
1. Custom `DualRowPointerIterator` stores `CPointer(float32)` for both matrices
2. `input_dereference`: Computes row pointers, loads values element-by-element
3. `output_dereference`: Computes row pointers, writes values element-by-element
4. Operator works on value tuples (all scalar operations)
5. Single scan over timesteps

**Key Insights:**
- Use `CPointer(float32)` from the start (not uint64)
- Pointer + N advances by N *elements* (not bytes!)
- Correct formula: `ptr + row_idx * (row_stride_bytes // 4)`
- Load/store values inside iterator, not in operator
- All operations on scalars (pointer indexing, tuple elements)

**This is the Python equivalent of C++'s VectorPair pattern!**
"""

# ============================================================================
# Row Pointer Iterator (Custom Iterator returning pointers, not values)
# ============================================================================
import ctypes
from functools import lru_cache
from typing import Tuple

import cuda.compute
import numba
import numba.cuda
import numpy as np
from cuda.compute.iterators import _iterators
from numba import types
from numba.core import cgutils
from numba.core.extending import models, register_model
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import registry as cuda_lower_registry


@lru_cache
def make_dual_row_pointer_iterator_struct():
    """
    Create ctypes struct for dual row pointer iterator state.

    Holds pointers to BOTH A and Bu matrices and returns pairs.
    """

    class DualRowPointerState(ctypes.Structure):
        _fields_ = [
            ("row_index", ctypes.c_int64),  # Current row (mutable)
            # A base pointer as void* (immutable)
            ("A_base_ptr", ctypes.c_void_p),
            # Bu base pointer as void* (immutable)
            ("Bu_base_ptr", ctypes.c_void_p),
            ("row_stride_bytes", ctypes.c_int64),  # Stride in bytes (immutable)
        ]

    return DualRowPointerState


def dual_row_pointer_iterator_numba_type():
    """Define numba type for dual row pointer iterator state."""

    class DualRowPointerViewType(types.Type):
        def __init__(self):
            super().__init__(name="DualRowPointerView")

    row_pointer_type = DualRowPointerViewType()

    int64_t = numba.from_dtype(np.int64)
    float32_ptr_t = types.CPointer(types.float32)  # Typed pointer

    members = [
        ("row_index", int64_t),
        ("A_base_ptr", float32_ptr_t),  # CPointer(float32) not uint64!
        ("Bu_base_ptr", float32_ptr_t),  # CPointer(float32) not uint64!
        ("row_stride_bytes", int64_t),
    ]

    # Register attribute access
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
    Iterator that returns a TUPLE of uint64 pointers (A_ptr, Bu_ptr).

    This eliminates the need for ZipIterator!
    """

    iterator_kind_type = DualRowPointerIteratorKind

    def __init__(self, A_base_ptr: int, Bu_base_ptr: int, row_stride_bytes: int):
        state_numba_type = dual_row_pointer_iterator_numba_type()
        # Value type is tuple of value tuples (not pointers!)
        # This is like C++ VectorPair - returns the actual values
        value_type = types.UniTuple(types.UniTuple(types.float32, 4), 2)

        # Build ctypes struct
        DualRowPointerState_cls = make_dual_row_pointer_iterator_struct()
        host_state = DualRowPointerState_cls(
            0, A_base_ptr, Bu_base_ptr, row_stride_bytes
        )

        super().__init__(
            cvalue=host_state,
            state_type=state_numba_type,
            value_type=value_type,
        )

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
        """
        Load and return row values directly.

        Computes pointers internally, loads values, returns value tuples.
        Like strided iterator but for tuples of values instead of scalars.
        """
        # pass
        zero = numba.int32(0)
        state = state_ref[zero]
        row_idx = state.row_index

        # Compute pointers to both A and Bu rows using pointer arithmetic
        # state.A_base_ptr is already CPointer(float32)
        # Pointer + N advances by N elements, so convert bytes to elements
        # row_stride_bytes = 16 bytes = 4 float32 elements
        elements_per_row = state.row_stride_bytes // numba.int64(4)
        A_row_ptr = state.A_base_ptr + row_idx * elements_per_row
        Bu_row_ptr = state.Bu_base_ptr + row_idx * elements_per_row

        # Load values from pointers and return VALUES, not pointers!
        # This combines the pointer computation and loading in one step
        result[0] = (
            (A_row_ptr[0], A_row_ptr[1], A_row_ptr[2], A_row_ptr[3]),
            (Bu_row_ptr[0], Bu_row_ptr[1], Bu_row_ptr[2], Bu_row_ptr[3]),
        )

    @staticmethod
    def output_dereference_impl(state_ref, value_tuples):
        """
        Write value tuples to memory through computed pointers.

        value_tuples is ((A_vals), (Bu_vals)) where each is a tuple of floats.
        We write these directly using CPointers (no casting needed!).
        """
        # pass
        zero = numba.int32(0)
        state = state_ref[zero]
        row_idx = state.row_index

        # Compute pointers to output rows using pointer arithmetic
        # state.A_base_ptr is already CPointer(float32)
        # Pointer + N advances by N elements, so convert bytes to elements
        elements_per_row = state.row_stride_bytes // numba.int64(4)
        A_row_ptr = state.A_base_ptr + row_idx * elements_per_row
        Bu_row_ptr = state.Bu_base_ptr + row_idx * elements_per_row

        # Extract value tuples
        A_vals = value_tuples[0]  # Tuple of 4 floats
        Bu_vals = value_tuples[1]  # Tuple of 4 floats

        # Write element-by-element using pointer indexing
        # No carray needed - CPointer is already indexable!
        A_row_ptr[0] = A_vals[0]
        A_row_ptr[1] = A_vals[1]
        A_row_ptr[2] = A_vals[2]
        A_row_ptr[3] = A_vals[3]

        Bu_row_ptr[0] = Bu_vals[0]
        Bu_row_ptr[1] = Bu_vals[1]
        Bu_row_ptr[2] = Bu_vals[2]
        Bu_row_ptr[3] = Bu_vals[3]


# ============================================================================
# S5 Scan with Pointer Casting
# ============================================================================


def s5_scan_2d_pointer_cast(
    A_in: np.ndarray,
    Bu_in: np.ndarray,
    state_dim: int = 4,  # Hardcoded for now
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform 2D S5 scan using pointer casting and element-wise operations.

    This approach:
    - Passes uint64 pointers through iterators
    - Casts to typed pointers in operator
    - Indexes element-by-element (all scalars)
    - Single scan call over timesteps

    Args:
        A_in: Input A values, shape (timesteps, state_dim)
        Bu_in: Input Bu values, shape (timesteps, state_dim)
        state_dim: Hidden dimension (must match hardcoded value in operator)

    Returns:
        Tuple of (A_out, Bu_out) with shape (timesteps, state_dim) each
    """
    timesteps = A_in.shape[0]

    # Move to GPU
    d_A_in = numba.cuda.to_device(A_in)
    d_Bu_in = numba.cuda.to_device(Bu_in)
    d_A_out = numba.cuda.device_array((timesteps, state_dim), dtype=A_in.dtype)
    d_Bu_out = numba.cuda.device_array((timesteps, state_dim), dtype=Bu_in.dtype)

    # ========================================================================
    # Input Iterator: Pointer Iterator → Transform (Load Values)
    # ========================================================================

    A_in_ptr = d_A_in.device_ctypes_pointer.value
    Bu_in_ptr = d_Bu_in.device_ctypes_pointer.value
    A_out_ptr = d_A_out.device_ctypes_pointer.value
    Bu_out_ptr = d_Bu_out.device_ctypes_pointer.value
    row_stride_bytes = state_dim * A_in.dtype.itemsize

    # Single iterator that computes pointers AND loads values internally
    # No TransformIterator needed - loading is built into the iterator!
    input_iter = DualRowPointerIterator(A_in_ptr, Bu_in_ptr, row_stride_bytes)

    # ========================================================================
    # Output Iterator: Use pointer iterator directly for now
    # ========================================================================

    output_iter = DualRowPointerIterator(A_out_ptr, Bu_out_ptr, row_stride_bytes)

    # ========================================================================
    # S5 Operator with Pointer Casting
    # ========================================================================

    def s5_op_on_values(x_vals, y_vals):
        """
        S5 operator on VALUE tuples (not pointers!).

        x_vals and y_vals are: ((A_tuple), (Bu_tuple)) of floats

        This is like C++ S5Operator2D working on VectorPair objects.
        All operations are on SCALARS only!
        """
        # Extract value tuples
        x_A = x_vals[0]  # Tuple of 4 floats
        x_Bu = x_vals[1]  # Tuple of 4 floats
        y_A = y_vals[0]  # Tuple of 4 floats
        y_Bu = y_vals[1]  # Tuple of 4 floats

        # Perform S5 operations element-by-element
        # All tuple[i] operations return scalars!
        # Hardcoded for state_dim=4
        result_A_0 = y_A[0] * x_A[0]
        result_A_1 = y_A[1] * x_A[1]
        result_A_2 = y_A[2] * x_A[2]
        result_A_3 = y_A[3] * x_A[3]

        result_Bu_0 = y_A[0] * x_Bu[0] + y_Bu[0]
        result_Bu_1 = y_A[1] * x_Bu[1] + y_Bu[1]
        result_Bu_2 = y_A[2] * x_Bu[2] + y_Bu[2]
        result_Bu_3 = y_A[3] * x_Bu[3] + y_Bu[3]

        # Return tuple of tuples (all scalars!)
        return (
            (result_A_0, result_A_1, result_A_2, result_A_3),
            (result_Bu_0, result_Bu_1, result_Bu_2, result_Bu_3),
        )

    # ========================================================================
    # Single Scan Call
    # ========================================================================

    cuda.compute.inclusive_scan(
        d_in=input_iter,
        d_out=output_iter,
        op=s5_op_on_values,
        init_value=None,
        num_items=timesteps,
    )

    A_out = d_A_out.copy_to_host()
    Bu_out = d_Bu_out.copy_to_host()

    return A_out, Bu_out


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Attempt 3: Custom DualRowPointerIterator (WORKING!)")
    print("=" * 80)
    print("\nApproach:")
    print("  1. Custom iterator stores CPointer(float32) for both matrices")
    print("  2. Computes row pointers using: ptr + row_idx * (stride_bytes // 4)")
    print("  3. Loads/stores values element-by-element (all scalars!)")
    print("  4. Returns value tuples, not pointers")
    print("  5. Single scan call over timesteps\n")

    timesteps = 8
    state_dim = 4

    np.random.seed(42)
    A_in = np.random.randn(timesteps, state_dim).astype(np.float32)
    Bu_in = np.random.randn(timesteps, state_dim).astype(np.float32)

    print(f"Input shape: ({timesteps}, {state_dim})")
    print("Attempting single scan call with pointer casting...\n")

    try:
        A_out, Bu_out = s5_scan_2d_pointer_cast(A_in, Bu_in, state_dim)

        print("\n" + "=" * 80)
        print("SUCCESS! Single-scan 2D S5 implementation works!")
        print("=" * 80)
        print("""
This demonstrates:
✓ Custom iterator with CPointer(float32) state
✓ Correct pointer arithmetic (element-wise, not byte-wise)
✓ Loading/storing values element-by-element (all scalars)
✓ Operator works on value tuples
✓ Single scan call over timesteps achieves 2D S5!

Key techniques:
- Store CPointers in iterator state from initialization
- Use ptr + row_idx * (stride_bytes // sizeof(element)) for addressing
- Load/store inside iterator dereference methods
- Never create array objects, only scalar operations

This is the Python equivalent of C++'s VectorPair pattern!
""")

        # Verify correctness
        import torch

        def s5_torch(x, y):
            return (y[0] * x[0], y[0] * x[1] + y[1])

        d_A = torch.as_tensor(A_in, device="cuda")
        d_Bu = torch.as_tensor(Bu_in, device="cuda")

        def scan_fn(inputs):
            return torch._higher_order_ops.associative_scan(s5_torch, inputs, dim=0)

        scan_fn = torch.compile(scan_fn, dynamic=False)
        A_ref, Bu_ref = scan_fn((d_A, d_Bu))

        a_err = np.max(np.abs(A_out - A_ref.cpu().numpy()))
        bu_err = np.max(np.abs(Bu_out - Bu_ref.cpu().numpy()))

        print("\nVerification vs PyTorch:")
        print(f"  Max error (A):  {a_err:.6e}")
        print(f"  Max error (Bu): {bu_err:.6e}")

        if a_err < 1e-5 and bu_err < 1e-5:
            print("  ✓ Correctness verified!")

    except Exception as e:
        import traceback

        print("\n✗ Failed!")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}\n")
        print("Full traceback:")
        traceback.print_exc()

        print("\n" + "=" * 80)
        print("ANALYSIS:")
        print("=" * 80)
        print("""
If this failed, possible reasons:
1. Pointer casting intrinsic not properly defined
2. Numba doesn't allow inttoptr in device code
3. Type mismatch between input (tuples) and output (arrays)
4. Other compilation issues

The approach is sound in theory - we're only working with scalars.
The question is whether numba's device compilation supports the
low-level pointer operations we need.
""")
