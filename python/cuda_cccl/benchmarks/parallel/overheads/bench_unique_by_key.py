import sys

import numpy as np
import torch

import cuda.cccl.parallel.experimental.algorithms as algorithms
import cuda.nvbench as nvbench


def unique_by_key(state: nvbench.State):
    "Benchmark unique_by_key example"
    n_elems = state.getInt64("numElems")

    d_in_keys = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_in_items = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_out_keys = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_out_items = torch.rand(n_elems, dtype=torch.float32, device="cuda")
    d_out_num_selected = torch.empty(1, dtype=torch.int32, device="cuda")

    def compare_op(lhs, rhs):
        return np.uint8(lhs == rhs)

    alg = algorithms.unique_by_key(
        d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected, compare_op
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(
        None,
        d_in_keys,
        d_in_items,
        d_out_keys,
        d_out_items,
        d_out_num_selected,
        n_elems,
    )

    temp_storage = torch.empty(temp_nbytes, dtype=torch.uint8, device="cuda")

    torch.cuda.synchronize()

    def launcher(launch: nvbench.Launch):
        alg(
            temp_storage,
            d_in_keys,
            d_in_items,
            d_out_keys,
            d_out_items,
            d_out_num_selected,
            n_elems,
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(unique_by_key)
    b.addInt64Axis("numElems", [2**20, 2**26])

    nvbench.run_all_benchmarks(sys.argv)
