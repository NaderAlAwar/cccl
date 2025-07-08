import sys

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.nvbench as nvbench


def merge_sort(state: nvbench.State):
    "Benchmark segmented_reduce example"
    n_elems = state.getInt64("numElems")

    state.add_summary("numElems", n_elems)
    state.collectCUPTIMetrics()

    rng = cp.random.default_rng()
    d_in_keys = rng.random(n_elems, dtype=cp.float32)
    d_in_items = cp.ones(n_elems, dtype=cp.float32)
    d_out_keys = cp.empty(n_elems, dtype=cp.float32)
    d_out_items = cp.empty(n_elems, dtype=cp.float32)

    def compare_op(lhs, rhs):
        return np.uint8(lhs < rhs)

    alg = algorithms.merge_sort_low_overhead(
        d_in_keys, d_in_items, d_out_keys, d_out_items, compare_op
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(
        None,
        0,
        d_in_keys.data.ptr,
        d_in_items.data.ptr,
        d_out_keys.data.ptr,
        d_out_items.data.ptr,
        n_elems,
    )

    temp_storage = cp.empty(temp_nbytes, dtype=cp.uint8)

    def launcher(launch: nvbench.Launch):
        alg(
            temp_storage.data.ptr,
            temp_storage.nbytes,
            d_in_keys.data.ptr,
            d_in_items.data.ptr,
            d_out_keys.data.ptr,
            d_out_items.data.ptr,
            n_elems,
        )
        cp.cuda.runtime.deviceSynchronize()

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    b = nvbench.register(merge_sort)
    b.addInt64Axis("numElems", [2**20, 2**22, 2**24])

    nvbench.run_all_benchmarks(sys.argv)
