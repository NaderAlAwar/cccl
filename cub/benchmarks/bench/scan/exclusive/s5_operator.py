import sys

import cuda.bench as bench
import torch
from torch._higher_order_ops import associative_scan

# Increase recompile limit to avoid warnings when benchmarking different tensor sizes
torch._dynamo.config.recompile_limit = 100


def as_torch_cuda_Stream(
    cs: bench.CudaStream, dev: int | None
) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def s5_operator(x, y):
    A_i, Bu_i = x
    A_j, Bu_j = y
    return (A_j * A_i, A_j * Bu_i + Bu_j)


def s5_associative_scan(state: bench.State):
    """Benchmark different scans with different data types"""
    dtype_str = state.get_string("dtype")

    timesteps = state.get_int64("Timesteps")
    state_dim = 4  # hidden dimension

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[dtype_str]

    state.add_summary("dtype", dtype_str)

    # Create input tensor on CUDA
    dev_id = state.get_device()
    device = torch.device(f"cuda:{dev_id}")
    tc_s = as_torch_cuda_Stream(state.get_stream(), dev_id)

    with torch.cuda.device(device), torch.cuda.stream(tc_s):
        A_in = torch.randn(
            timesteps, state_dim, dtype=dtype, device=device
        ).contiguous()
        Bu_in = torch.randn(
            timesteps, state_dim, dtype=dtype, device=device
        ).contiguous()

    def scan_fn(tensor):
        return associative_scan(
            combine_fn=s5_operator,
            xs=tensor,
            dim=0,
            reverse=False,
            combine_mode="pointwise",
        )

    scan_fn = torch.compile(scan_fn, dynamic=False)

    torch.cuda.synchronize()

    def launcher(launch: bench.Launch):
        tc_s = as_torch_cuda_Stream(launch.get_stream(), dev_id)
        with torch.cuda.device(device), torch.cuda.stream(tc_s):
            scan_fn((A_in, Bu_in))

    # Use sync=True since cumulative ops synchronize internally
    state.exec(launcher, sync=True)


if __name__ == "__main__":
    # Register the benchmark
    b = bench.register(s5_associative_scan)

    # b.add_string_axis("dtype", ["float16", "float32", "float64"])
    b.add_string_axis("dtype", ["float16", "float32", "float64"])
    b.add_int64_power_of_two_axis("Timesteps", range(12, 25, 2))

    bench.run_all_benchmarks(sys.argv)
