import sys

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
    d_output = torch.empty(1, dtype=torch.float32, device="cuda")

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype=np.float32)

    alg = algorithms.reduce_into_low_overhead(d_input, d_output, add_op, h_init)

    # query size of temporary storage and allocate
    temp_nbytes = alg(None, 0, d_input.data_ptr(), d_output.data_ptr(), n_elems, h_init)

    temp_storage = torch.empty(temp_nbytes, dtype=torch.uint8, device="cuda")

    torch.cuda.synchronize()

    def launcher(launch: nvbench.Launch):
        torch.cuda.synchronize()
        alg(
            temp_storage.data_ptr(),
            temp_storage.nbytes,
            d_input.data_ptr(),
            d_output.data_ptr(),
            n_elems,
            h_init,
        )
        torch.cuda.synchronize()

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    b = nvbench.register(reduce_into)
    b.addInt64Axis("numElems", [2**20, 2**26])

    nvbench.run_all_benchmarks(sys.argv)
