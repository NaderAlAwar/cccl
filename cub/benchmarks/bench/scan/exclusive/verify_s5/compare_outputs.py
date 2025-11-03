#!/usr/bin/env python3
"""
Compare outputs from C++ and Python S5 scan implementations
"""

import argparse

import numpy as np


def compare_arrays(arr1, arr2, name, rtol=1e-5, atol=1e-8):
    """
    Compare two numpy arrays with appropriate tolerances

    Args:
        arr1: First array (e.g., C++ output)
        arr2: Second array (e.g., Python output)
        name: Name of the arrays being compared (for printing)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if arrays match within tolerance, False otherwise
    """
    if arr1.shape != arr2.shape:
        print(f"❌ {name}: Shape mismatch! {arr1.shape} vs {arr2.shape}")
        return False

    # Check for NaN or Inf
    if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
        print(f"❌ {name}: Contains NaN values!")
        return False

    if np.any(np.isinf(arr1)) or np.any(np.isinf(arr2)):
        print(f"❌ {name}: Contains Inf values!")
        return False

    # Compute differences
    abs_diff = np.abs(arr1 - arr2)
    rel_diff = abs_diff / (np.abs(arr2) + 1e-10)  # Avoid division by zero

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)

    # Check if arrays are close
    is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)

    print(f"\n{name}:")
    print(f"  Shape: {arr1.shape}")
    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"  Max relative difference: {max_rel_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")

    if is_close:
        print(f"  ✅ Arrays match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"  ❌ Arrays do NOT match within tolerance (rtol={rtol}, atol={atol})")

        # Find locations with largest differences
        diff_indices = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"  Largest difference at index {diff_indices}:")
        print(f"    C++:    {arr1[diff_indices]}")
        print(f"    Python: {arr2[diff_indices]}")

    return is_close


def main():
    parser = argparse.ArgumentParser(
        description="Compare C++ and Python S5 scan outputs"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-4, help="Relative tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-6, help="Absolute tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type (affects tolerance defaults)",
    )

    args = parser.parse_args()

    # Adjust tolerances based on dtype (only if user didn't specify custom values)
    if args.dtype == "float16":
        rtol = args.rtol if args.rtol != 1e-4 else 1e-3  # Lower precision for fp16
        atol = args.atol if args.atol != 1e-6 else 1e-5
    elif args.dtype == "float32":
        rtol = args.rtol  # Use default 1e-4
        atol = args.atol  # Use default 1e-6
    else:  # float64
        rtol = args.rtol if args.rtol != 1e-4 else 1e-7  # Higher precision for fp64
        atol = args.atol if args.atol != 1e-6 else 1e-10

    print("=" * 70)
    print("Comparing C++ and Python S5 Scan Outputs")
    print("=" * 70)
    print(f"Tolerances: rtol={rtol}, atol={atol}")

    try:
        # Load C++ outputs
        A_out_cpp = np.load("A_out_cpp.npy")
        Bu_out_cpp = np.load("Bu_out_cpp.npy")

        # Load Python outputs
        A_out_python = np.load("A_out_python.npy")
        Bu_out_python = np.load("Bu_out_python.npy")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find output file: {e}")
        print("\nMake sure to run both verification programs first:")
        print(
            "  ./s5_verify --timesteps N --is-2d [0|1] --dtype [float16|float32|float64]"
        )
        print(
            "  python s5_verify.py --timesteps N --is-2d [0|1] --dtype [float16|float32|float64]"
        )
        return 1

    # Compare A outputs
    a_match = compare_arrays(A_out_cpp, A_out_python, "A output", rtol=rtol, atol=atol)

    # Compare Bu outputs
    bu_match = compare_arrays(
        Bu_out_cpp, Bu_out_python, "Bu output", rtol=rtol, atol=atol
    )

    # Final result
    print("\n" + "=" * 70)
    if a_match and bu_match:
        print("✅ SUCCESS: All outputs match!")
        print("=" * 70)
        return 0
    else:
        print("❌ FAILURE: Outputs do not match!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
