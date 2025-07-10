import time

import numpy as np
import torch

import cuda.cccl.parallel.experimental.algorithms as algorithms


def reduce_into():
    n_elems = 2**20

    d_input = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_output = torch.empty(1, dtype=torch.float32, device="cuda")

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype=np.float32)

    alg = algorithms.reduce_into(d_input, d_output, add_op, h_init)

    # query size of temporary storage and allocate
    temp_nbytes = alg(None, d_input, d_output, n_elems, h_init)

    temp_storage = torch.empty(temp_nbytes, dtype=torch.uint8, device="cuda")

    for _ in range(10):
        alg(temp_storage, d_input, d_output, n_elems, h_init)

    torch.cuda.synchronize()

    execution_times = []
    for _ in range(30):
        start = time.perf_counter_ns()
        alg(temp_storage, d_input, d_output, n_elems, h_init)
        stop = time.perf_counter_ns()
        execution_times.append(stop - start)

    avg_time_ns = sum(execution_times) / len(execution_times)
    print(f"Num elems {n_elems}; Average execution time: {avg_time_ns:.2f} ns")


if __name__ == "__main__":
    reduce_into()
