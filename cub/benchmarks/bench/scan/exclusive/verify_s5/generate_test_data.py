#!/usr/bin/env python3
"""
Generate test data for S5 scan verification.
This script creates .npy files with input data that can be read by both
the C++ and Python implementations for output comparison.
"""

import argparse

import numpy as np


def generate_test_data(timesteps, state_dim, dtype, is_2d, seed=42):
    """
    Generate test data for S5 scan verification.

    Args:
        timesteps: Number of timesteps (sequence length)
        state_dim: Hidden dimension size
        dtype: numpy dtype (e.g., np.float32, np.float16, np.float64)
        is_2d: If True, generate 2D tensors, otherwise 1D scalars
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    if is_2d:
        # Generate 2D tensors: (timesteps, state_dim)
        A_in = np.random.randn(timesteps, state_dim).astype(dtype)
        Bu_in = np.random.randn(timesteps, state_dim).astype(dtype)
        print(
            f"Generated 2D data: A_in shape={A_in.shape}, Bu_in shape={Bu_in.shape}, dtype={dtype}"
        )
    else:
        # Generate 1D scalars: (timesteps,)
        A_in = np.random.randn(timesteps).astype(dtype)
        Bu_in = np.random.randn(timesteps).astype(dtype)
        print(
            f"Generated 1D data: A_in shape={A_in.shape}, Bu_in shape={Bu_in.shape}, dtype={dtype}"
        )

    return A_in, Bu_in


def save_test_data(A_in, Bu_in, output_dir="."):
    """Save test data to .npy files"""
    np.save(f"{output_dir}/A_in.npy", A_in)
    np.save(f"{output_dir}/Bu_in.npy", Bu_in)
    print(f"Saved to {output_dir}/A_in.npy and {output_dir}/Bu_in.npy")


def main():
    parser = argparse.ArgumentParser(description="Generate S5 scan test data")
    parser.add_argument(
        "--timesteps", type=int, default=1024, help="Number of timesteps"
    )
    parser.add_argument(
        "--state-dim", type=int, default=40, help="Hidden dimension size"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--is-2d",
        type=int,
        default=1,
        choices=[0, 1],
        help="Generate 2D (1) or 1D (0) data",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output directory for .npy files"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Map dtype string to numpy dtype
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    dtype = dtype_map[args.dtype]

    # Generate and save data
    A_in, Bu_in = generate_test_data(
        args.timesteps, args.state_dim, dtype, bool(args.is_2d), args.seed
    )
    save_test_data(A_in, Bu_in, args.output_dir)

    print("\nTo run verification:")
    print(
        f"  C++:    ./s5_verify --timesteps {args.timesteps} --is-2d {args.is_2d} --dtype {args.dtype}"
    )
    print(
        f"  Python: python s5_verify.py --timesteps {args.timesteps} --is-2d {args.is_2d} --dtype {args.dtype}"
    )


if __name__ == "__main__":
    main()
