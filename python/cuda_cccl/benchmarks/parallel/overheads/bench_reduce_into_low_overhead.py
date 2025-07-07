import sys

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.nvbench as nvbench


def reduce_into(state: nvbench.State):
    "Benchmark segmented_reduce example"
    n_elems = state.getInt64("numElems")

    state.add_summary("numElemes", n_elems)
    state.collectCUPTIMetrics()

    rng = cp.random.default_rng()
    d_input = rng.random(n_elems, dtype=cp.float32)
    d_output = cp.empty(1, dtype=cp.float32)

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype=np.float32)

    alg = algorithms.reduce_into_low_overhead(d_input, d_output, add_op, h_init)

    # query size of temporary storage and allocate
    temp_nbytes = alg(None, 0, d_input.data.ptr, d_output.data.ptr, n_elems, h_init)

    temp_storage = cp.empty(temp_nbytes, dtype=cp.uint8)

    def launcher(launch: nvbench.Launch):
        alg(
            temp_storage.data.ptr,
            temp_storage.nbytes,
            d_input.data.ptr,
            d_output.data.ptr,
            n_elems,
            h_init,
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(reduce_into)
    b.addInt64Axis("numElems", [2**20, 2**22, 2**24])

    nvbench.run_all_benchmarks(sys.argv)
