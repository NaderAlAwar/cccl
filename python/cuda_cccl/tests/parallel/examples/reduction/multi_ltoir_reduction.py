# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Propagate multiple device ops' LTO-IRs through the Op.extra_ltoirs path.

For now, we simulate the future compile_all() interface by compiling
additional ops individually and passing them as extra_ltoirs.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel
from cuda.cccl.parallel.experimental import _cccl_interop as cccl
from cuda.cccl.parallel.experimental._bindings import Op as _OpBinding
from cuda.cccl.parallel.experimental.op import OpKind

# Prepare input/output
dtype = np.float32
h_init = np.array([0], dtype=dtype)
d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
d_output = cp.empty(1, dtype=dtype)


def add_op(a, b):
    return a + b


def mul_op(a, b):
    return a * b


def max_op(a, b):
    return a if a > b else b


def min_op(a, b):
    return a if a < b else b


_orig_to_cccl_op = cccl.to_cccl_op


def _patched_to_cccl_op(op, sig):
    # Preserve behavior for known OpKind
    if isinstance(op, OpKind):
        return _orig_to_cccl_op(op, sig)

    # Wrap the primary op (e.g., add_op)
    wrapped_op, wrapper_sig = cccl._create_void_ptr_wrapper(op, sig)

    # Desired future API:
    # from numba import cuda
    # ltoir, extra_ltoirs = cuda.compile_all(wrapped_op, sig=wrapper_sig, output="ltoir")

    # Current workaround: compile additional ops one-by-one and pass them as extras
    # Only the main op is actually used in the reduce; extras are propagated.
    from numba import cuda  # local import to keep example self-contained

    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")

    extra_ops = [mul_op, max_op, min_op]
    extra_ltoirs = []
    for extra in extra_ops:
        w_extra, sig_extra = cccl._create_void_ptr_wrapper(extra, sig)
        lto_extra, _ = cuda.compile(w_extra, sig=sig_extra, output="ltoir")
        extra_ltoirs.append(lto_extra)

    return _OpBinding(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
        extra_ltoirs=extra_ltoirs,
    )


# Monkeypatch just for this example run
cccl.to_cccl_op = _patched_to_cccl_op  # type: ignore[attr-defined]

# Perform the reduction (only add_op semantics are used)
parallel.reduce_into(d_input, d_output, add_op, len(d_input), h_init)

# Verify the result
expected_output = 15
assert (d_output == expected_output).all()
print(f"Sum reduction result with extra LTO-IRs propagated: {d_output[0]}")
# example-end
