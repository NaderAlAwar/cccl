import sys

import numpy as np
import torch

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.nvbench as nvbench


def reduce_into(state: nvbench.State):
    "Benchmark segmented_reduce example"
    n_elems = state.getInt64("numElems")

    d_input = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_output = torch.empty(1, dtype=torch.float32, device="cuda")

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype=np.float32)

    alg = algorithms.reduce_into(d_input, d_output, add_op, h_init)

    # query size of temporary storage and allocate
    temp_nbytes = alg(None, d_input, d_output, n_elems, h_init)

    temp_storage = torch.empty(temp_nbytes, dtype=torch.uint8, device="cuda")

    torch.cuda.synchronize()

    def launcher(launch: nvbench.Launch):
        alg(temp_storage, d_input, d_output, n_elems, h_init)

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(reduce_into)
    b.addInt64Axis("numElems", [2**20, 2**26])

    nvbench.run_all_benchmarks(sys.argv)
