import sys
import time

import cupy as cp
import numpy as np
import torch

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.nvbench as nvbench


def reduce_into(state: nvbench.State):
    "Benchmark segmented_reduce example"
    n_elems = state.getInt64("numElems")

    state.add_summary("numElemes", n_elems)
    state.collectCUPTIMetrics()

    d_input = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_output = torch.empty(1, dtype=torch.float32)

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype=np.float32)

    alg = algorithms.reduce_into_low_overhead(d_input, d_output, add_op, h_init)

    # query size of temporary storage and allocate
    temp_nbytes = alg(None, 0, d_input.data.ptr, d_output.data.ptr, n_elems, h_init)

    temp_storage = torch.empty(temp_nbytes, dtype=torch.uint8)

    for _ in range(10):
        alg(
            temp_storage.data.ptr,
            temp_storage.nbytes,
            d_input.data.ptr,
            d_output.data.ptr,
            n_elems,
            h_init,
        )

    cp.cuda.runtime.deviceSynchronize()
    execution_times = []
    for _ in range(100):
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter_ns()
        alg(
            temp_storage.data.ptr,
            temp_storage.nbytes,
            d_input.data.ptr,
            d_output.data.ptr,
            n_elems,
            h_init,
        )
        cp.cuda.runtime.deviceSynchronize()
        stop = time.perf_counter_ns()
        execution_times.append(stop - start)

    avg_time_ns = sum(execution_times) / len(execution_times)
    print(f"Num elems {n_elems}; Average execution time: {avg_time_ns:.2f} ns")


if __name__ == "__main__":
    b = nvbench.register(reduce_into)
    b.addInt64Axis("numElems", [2**20, 2**26])

    nvbench.run_all_benchmarks(sys.argv)
