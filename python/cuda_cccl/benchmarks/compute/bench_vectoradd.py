import sys

import torch

import cuda.bench as bench
import cuda.compute
import cuda.core.experimental as core
from cuda.compute import OpKind


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


def vectoradd(state: bench.State) -> None:
    build_A = torch.empty(2, 2, dtype=torch.float16, device="cuda").contiguous()
    build_B = torch.empty(2, 2, dtype=torch.float16, device="cuda").contiguous()
    build_C = torch.empty(2, 2, dtype=torch.float16, device="cuda").contiguous()

    dev_id = state.get_device()
    tc_s = as_torch_cuda_Stream(state.get_stream(), dev_id)

    size = 16384

    with tc_s:
        seed = 42
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        A = torch.randn(
            size, size, dtype=torch.float16, device="cuda", generator=gen
        ).contiguous()
        B = torch.randn(
            size, size, dtype=torch.float16, device="cuda", generator=gen
        ).contiguous()
        C = torch.empty(size, size, dtype=torch.float16, device="cuda").contiguous()

    transformer = cuda.compute.make_binary_transform(
        build_A, build_B, build_C, OpKind.PLUS
    )

    def launcher(launch: bench.Launch) -> None:
        s = as_cccl_Stream(launch.get_stream())
        size = len(A)
        transformer(A, B, C, size * size, s)

    state.exec(launcher)


if __name__ == "__main__":
    b = bench.register(vectoradd)
    b.add_int64_axis("Elements{io}", [2**28])

    bench.run_all_benchmarks(sys.argv)
