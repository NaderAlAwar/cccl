# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
import cupy as cp

from cuda.compute.algorithms import select_flagged

# Create input data and flags
d_in = cp.array([10, 20, 30, 40, 50, 60], dtype=cp.int32)
d_flags = cp.array([0, 1, 0, 1, 1, 0], dtype=cp.int32)
d_out = cp.empty_like(d_in)
d_num_selected = cp.zeros(2, dtype=cp.uint64)


# Select values whose corresponding flag is nonzero
def is_selected(flag):
    return flag != 0


# Execute flagged select
select_flagged(d_in, d_flags, d_out, d_num_selected, is_selected, len(d_in))

# Get results
num_selected = int(d_num_selected[0])
result = d_out[:num_selected].get()
print(f"Selected {num_selected} items: {result}")
# Output: Selected 3 items: [20 40 50]
# example-end

assert num_selected == 3
assert (result == [20, 40, 50]).all()
