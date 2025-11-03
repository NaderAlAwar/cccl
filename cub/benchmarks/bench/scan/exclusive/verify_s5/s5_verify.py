#!/usr/bin/env python3
"""
S5 Associative Scan Verification Tool (Python/PyTorch)
Reads input from .npy files, runs scan, writes output to .npy files for comparison
"""

import argparse

import numpy as np
import torch
from torch._higher_order_ops import associative_scan


def s5_operator(x, y):
    """S5 associative scan operator"""
    A_i, Bu_i = x
    A_j, Bu_j = y
    return (A_j * A_i, A_j * Bu_i + Bu_j)


def run_verification_1d(timesteps, dtype):
    """Run 1D (scalar) verification"""
    print(f"Running 1D verification (timesteps={timesteps}, dtype={dtype})")

    # Read input data
    A_in = np.load("A_in.npy")
    Bu_in = np.load("Bu_in.npy")

    if A_in.shape != (timesteps,) or Bu_in.shape != (timesteps,):
        print(
            f"Error: Input shape mismatch. Expected ({timesteps},) but got A={A_in.shape}, Bu={Bu_in.shape}"
        )
        return

    # Convert to torch tensors
    device = torch.device("cuda:0")
    A_in_torch = torch.from_numpy(A_in).to(device=device, dtype=dtype)
    Bu_in_torch = torch.from_numpy(Bu_in).to(device=device, dtype=dtype)

    # Run scan
    A_out_torch, Bu_out_torch = associative_scan(
        combine_fn=s5_operator,
        xs=(A_in_torch, Bu_in_torch),
        dim=0,
        reverse=False,
        combine_mode="pointwise",
    )

    torch.cuda.synchronize()

    # Convert back to numpy
    A_out = A_out_torch.cpu().numpy()
    Bu_out = Bu_out_torch.cpu().numpy()

    # Save output
    np.save("A_out_python.npy", A_out)
    np.save("Bu_out_python.npy", Bu_out)

    print("Output written to A_out_python.npy and Bu_out_python.npy")


def run_verification_2d(timesteps, state_dim, dtype):
    """Run 2D (vector) verification"""
    print(
        f"Running 2D verification (timesteps={timesteps}, state_dim={state_dim}, dtype={dtype})"
    )

    # Read input data
    A_in = np.load("A_in.npy")
    Bu_in = np.load("Bu_in.npy")

    expected_shape = (timesteps, state_dim)
    if A_in.shape != expected_shape or Bu_in.shape != expected_shape:
        print(
            f"Error: Input shape mismatch. Expected {expected_shape} but got A={A_in.shape}, Bu={Bu_in.shape}"
        )
        return

    # Convert to torch tensors
    device = torch.device("cuda:0")
    A_in_torch = torch.from_numpy(A_in).to(device=device, dtype=dtype)
    Bu_in_torch = torch.from_numpy(Bu_in).to(device=device, dtype=dtype)

    # Run scan
    A_out_torch, Bu_out_torch = associative_scan(
        combine_fn=s5_operator,
        xs=(A_in_torch, Bu_in_torch),
        dim=0,
        reverse=False,
        combine_mode="pointwise",
    )

    torch.cuda.synchronize()

    # Convert back to numpy
    A_out = A_out_torch.cpu().numpy()
    Bu_out = Bu_out_torch.cpu().numpy()

    # Save output
    np.save("A_out_python.npy", A_out)
    np.save("Bu_out_python.npy", Bu_out)

    print("Output written to A_out_python.npy and Bu_out_python.npy")


def main():
    parser = argparse.ArgumentParser(description="S5 Scan Verification (Python)")
    parser.add_argument(
        "--timesteps", type=int, required=True, help="Number of timesteps"
    )
    parser.add_argument(
        "--is-2d",
        type=int,
        required=True,
        choices=[0, 1],
        help="1 for 2D (vector) scan, 0 for 1D (scalar) scan",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--state-dim", type=int, default=40, help="Hidden dimension size (for 2D case)"
    )

    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    torch_dtype = dtype_map[args.dtype]

    print("S5 Scan Verification (Python)")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Mode: {'2D' if args.is_2d else '1D'}")
    print(f"  Dtype: {args.dtype}")

    if args.is_2d:
        run_verification_2d(args.timesteps, args.state_dim, torch_dtype)
    else:
        run_verification_1d(args.timesteps, torch_dtype)


if __name__ == "__main__":
    main()
