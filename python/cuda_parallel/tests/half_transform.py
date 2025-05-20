import cupy as cp

import cuda.parallel.experimental.algorithms as algorithms


def binary_transform_device(d_input1, d_input2, d_output, num_items, op, stream=None):
    transform = algorithms.binary_transform(d_input1, d_input2, d_output, op)
    transform(d_input1, d_input2, d_output, num_items, stream=stream)


def op(a, b):
    return a + b


input_array = cp.ones(2**12, dtype=cp.float16)

d_in1 = input_array
d_in2 = input_array
d_out = cp.empty_like(d_in1)

binary_transform_device(d_in1, d_in2, d_out, len(d_in1), op)

cp.cuda.runtime.deviceSynchronize()
cp.set_printoptions(threshold=cp.inf)
print(d_out)
