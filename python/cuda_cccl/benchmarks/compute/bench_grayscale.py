import sys

import cupy as cp
import numpy as np
import torch

import cuda.bench as bench
import cuda.compute
import cuda.core.experimental as core
from cuda.compute import gpu_struct


class CCCLStream:
    "Class to work around https://github.com/NVIDIA/cccl/issues/5144"

    def __init__(self, ptr):
        self._ptr = ptr

    def __cuda_stream__(self):
        return (0, self._ptr)


def as_torch_cuda_Stream(
    cs: bench.CudaStream, dev: int | None
) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def as_core_Stream(cs: bench.CudaStream) -> core.Stream:
    return core.Stream.from_handle(cs.addressof())


def as_cccl_Stream(cs: bench.CudaStream) -> CCCLStream:
    return CCCLStream(cs.addressof())


@gpu_struct
class rgb_t:
    r: np.float32
    g: np.float32
    b: np.float32


def as_grayscale_op(pixel: rgb_t) -> np.float32:
    return (
        np.float32(0.2989) * pixel.r
        + np.float32(0.587) * pixel.g
        + np.float32(0.114) * pixel.b
    )


def grayscale(state: bench.State) -> None:
    # input_size = state.get_int64("Elements{io}")

    dev_id = state.get_device()
    tc_s = as_torch_cuda_Stream(state.get_stream(), dev_id)
    with torch.cuda.stream(tc_s):
        seed = 42
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        x = torch.rand(
            2**14, 2**14, 3, device="cuda", dtype=torch.float32, generator=gen
        ).contiguous()
        y = torch.empty(2**14, 2**14, device="cuda", dtype=torch.float32).contiguous()

    build_in = cp.empty(1, dtype=rgb_t)
    build_out = torch.empty(1, dtype=torch.float32, device="cuda")

    transformer = cuda.compute.make_unary_transform(
        build_in, build_out, as_grayscale_op
    )

    print(len(x))

    def launcher(launch: bench.Launch) -> None:
        s = as_cccl_Stream(launch.get_stream())
        size = len(x)
        transformer(x, y, size * size, s)

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(grayscale)
    b.add_int64_axis("Elements{io}", [2**28])

    bench.run_all_benchmarks(sys.argv)
