import sys
import time

import numpy as np
import torch

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.nvbench as nvbench


def merge_sort(state: nvbench.State):
    "Benchmark segmented_reduce example"
    n_elems = state.getInt64("numElems")

    state.add_summary("numElems", n_elems)
    state.collectCUPTIMetrics()

    d_in_keys = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_in_items = torch.ones(n_elems, dtype=torch.float32, device="cuda")
    d_out_keys = torch.empty(n_elems, dtype=torch.float32, device="cuda")
    d_out_items = torch.empty(n_elems, dtype=torch.float32, device="cuda")

    def compare_op(lhs, rhs):
        return np.uint8(lhs < rhs)

    alg = algorithms.merge_sort(
        d_in_keys, d_in_items, d_out_keys, d_out_items, compare_op
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(None, d_in_keys, d_in_items, d_out_keys, d_out_items, n_elems)

    temp_storage = torch.empty(temp_nbytes, dtype=torch.uint8, device="cuda")

    for _ in range(10):
        alg(temp_storage, d_in_keys, d_in_items, d_out_keys, d_out_items, n_elems)

    torch.cuda.synchronize()

    execution_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        alg(temp_storage, d_in_keys, d_in_items, d_out_keys, d_out_items, n_elems)
        torch.cuda.synchronize()
        stop = time.perf_counter_ns()
        execution_times.append(stop - start)

    avg_time_ns = sum(execution_times) / len(execution_times)
    print(f"Num elems {n_elems}; Average execution time: {avg_time_ns:.2f} ns")


if __name__ == "__main__":
    b = nvbench.register(merge_sort)
    b.addInt64Axis("numElems", [2**20, 2**22, 2**24])

    nvbench.run_all_benchmarks(sys.argv)
