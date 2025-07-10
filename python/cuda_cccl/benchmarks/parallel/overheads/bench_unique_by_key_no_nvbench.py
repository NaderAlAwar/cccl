import time

import numpy as np
import torch

import cuda.cccl.parallel.experimental.algorithms as algorithms


def unique_by_key():
    n_elems = 2**20

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

    for _ in range(10):
        alg(
            temp_storage,
            d_in_keys,
            d_in_items,
            d_out_keys,
            d_out_items,
            d_out_num_selected,
            n_elems,
        )

    torch.cuda.synchronize()

    execution_times = []
    for _ in range(30):
        start = time.perf_counter_ns()
        alg(
            temp_storage,
            d_in_keys,
            d_in_items,
            d_out_keys,
            d_out_items,
            d_out_num_selected,
            n_elems,
        )
        stop = time.perf_counter_ns()
        execution_times.append(stop - start)

    avg_time_ns = sum(execution_times) / len(execution_times)
    print(f"Num elems {n_elems}; Average execution time: {avg_time_ns:.2f} ns")


if __name__ == "__main__":
    unique_by_key()
