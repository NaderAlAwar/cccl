import sys

import numpy as np
import torch

import cuda.bench as bench
import cuda.compute


class CCCLStream:
    "Class to work around https://github.com/NVIDIA/cccl/issues/5144"

    def __init__(self, ptr):
        self._ptr = ptr

    def __cuda_stream__(self):
        return (0, self._ptr)


def as_cccl_Stream(cs: bench.CudaStream) -> CCCLStream:
    return CCCLStream(cs.addressof())


def histogram(state: bench.State) -> None:
    input_size = 10485760
    num_output_levels = np.array([257], dtype=np.int32)
    lower_level = np.array([0], dtype=np.int32)
    upper_level = np.array([256], dtype=np.int32)
    build_data = torch.randint(0, 256, (input_size,), dtype=torch.uint8, device="cuda")
    build_histogram = torch.zeros(
        (num_output_levels[0] - 1,), dtype=torch.uint64, device="cuda"
    )

    histogrammer = cuda.compute.make_histogram_even(
        build_data,
        build_histogram,
        num_output_levels,
        lower_level,
        upper_level,
        input_size,
    )

    temp_storage_size = histogrammer(
        None,
        build_data,
        build_histogram,
        num_output_levels,
        lower_level,
        upper_level,
        input_size,
    )
    d_temp_storage = torch.empty(temp_storage_size, dtype=torch.uint8, device="cuda")

    actual_histogram = torch.zeros(
        (num_output_levels[0] - 1,), dtype=torch.int64, device="cuda"
    )

    def launcher(launch: bench.Launch) -> None:
        s = as_cccl_Stream(launch.get_stream())
        histogrammer(
            d_temp_storage,
            build_data,
            actual_histogram,
            num_output_levels,
            lower_level,
            upper_level,
            input_size,
            s,
        )

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(histogram)
    b.add_int64_axis("Elements{io}", [10485760])
    b.add_int64_axis("Bins", [257])
    bench.run_all_benchmarks(sys.argv)
