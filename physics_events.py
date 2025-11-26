#!/usr/bin/env python3
"""
Utility script for generating benchmark input data and comparing benchmark outputs.

Usage:
  Generate raw input arrays with suffixes (saved as .npy files in the repo root):
      ./physics_events.py generate --seed 42

  Compare C++ vs Python invariant-mass results for all suffixes:
      ./physics_events.py compare --cpp-dir . --python-dir .

  Compare specific suffixes only:
      ./physics_events.py compare --suffixes _4 _5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path("/home/coder/cccl")


def _build_output_paths(base: Path, suffix: str = "") -> Dict[str, Path]:
    return {
        "electron_pts": base / f"electron_pts{suffix}.npy",
        "electron_etas": base / f"electron_etas{suffix}.npy",
        "electron_phis": base / f"electron_phis{suffix}.npy",
        "muons_pts": base / f"muons_pts{suffix}.npy",
        "muons_etas": base / f"muons_etas{suffix}.npy",
        "muons_phis": base / f"muons_phis{suffix}.npy",
        "electron_offsets": base / f"electron_offsets{suffix}.npy",
        "muon_offsets": base / f"muon_offsets{suffix}.npy",
    }


def generate_random_events(num_events: int, seed: int) -> Dict[str, np.ndarray]:
    """Replicates the NumPy data-generation snippet provided by the user."""
    rng = np.random.default_rng(seed)

    num_electrons_per_event = rng.integers(0, 11, size=num_events, dtype=np.int32)
    num_muons_per_event = rng.integers(0, 11, size=num_events, dtype=np.int32)

    electron_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int32),
            np.cumsum(num_electrons_per_event, dtype=np.int64).astype(np.int32),
        )
    )
    muon_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int32),
            np.cumsum(num_muons_per_event, dtype=np.int64).astype(np.int32),
        )
    )

    total_electrons = int(electron_offsets[-1])
    total_muons = int(muon_offsets[-1])

    electron_pts = rng.uniform(10, 100, size=total_electrons).astype(np.float64)
    electron_etas = rng.uniform(-3, 3, size=total_electrons).astype(np.float64)
    electron_phis = rng.uniform(0, 2 * np.pi, size=total_electrons).astype(np.float64)

    muon_pts = rng.uniform(10, 100, size=total_muons).astype(np.float64)
    muon_etas = rng.uniform(-3, 3, size=total_muons).astype(np.float64)
    muon_phis = rng.uniform(0, 2 * np.pi, size=total_muons).astype(np.float64)

    return {
        "electron_pts": electron_pts,
        "electron_etas": electron_etas,
        "electron_phis": electron_phis,
        "muons_pts": muon_pts,
        "muons_etas": muon_etas,
        "muons_phis": muon_phis,
        "electron_offsets": electron_offsets,
        "muon_offsets": muon_offsets,
        "num_electrons_per_event": num_electrons_per_event,
        "num_muons_per_event": num_muons_per_event,
    }


def save_generated_data(
    data: Dict[str, np.ndarray], output_dir: Path, suffix: str = ""
) -> None:
    output_paths = _build_output_paths(output_dir, suffix)
    for key, path in output_paths.items():
        np.save(path, data[key])
        print(f"Saved {key} -> {path}")

    total_electrons = int(data["electron_offsets"][-1])
    total_muons = int(data["muon_offsets"][-1])
    num_events = data["electron_offsets"].size - 1
    avg_electrons = total_electrons / max(num_events, 1)
    avg_muons = total_muons / max(num_events, 1)
    print(
        f"Generated {num_events:,} events | electrons: {total_electrons:,} "
        f"(avg {avg_electrons:.2f}) | muons: {total_muons:,} (avg {avg_muons:.2f})"
    )


def compare_outputs(
    cpp_dir: Path,
    python_dir: Path,
    atol: float,
    rtol: float,
    suffixes: list[str] = None,
) -> bool:
    """
    Compare C++ and Python output arrays.

    Args:
        cpp_dir: Directory containing C++ output files
        python_dir: Directory containing Python output files
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        suffixes: List of suffixes to compare (e.g., ['_4', '_5', '_6', '_7'])
                  If None, compares files without suffix

    Returns:
        True if all comparisons pass, False otherwise
    """
    if suffixes is None:
        suffixes = [""]

    targets = ("masses_electrons", "masses_muons")
    overall_success = True

    for suffix in suffixes:
        print(f"\n{'=' * 60}")
        print(f"Comparing outputs for suffix: '{suffix}'")
        print(f"{'=' * 60}")

        success_this_suffix = True
        for base in targets:
            cpp_path = cpp_dir / f"{base}{suffix}_cpp.npy"
            py_path = python_dir / f"{base}{suffix}_python.npy"

            if not cpp_path.exists():
                print(f"[ERROR] Missing C++ output: {cpp_path}", file=sys.stderr)
                success_this_suffix = False
                overall_success = False
                continue
            if not py_path.exists():
                print(f"[ERROR] Missing Python output: {py_path}", file=sys.stderr)
                success_this_suffix = False
                overall_success = False
                continue

            cpp_arr = np.load(cpp_path)
            py_arr = np.load(py_path)

            if cpp_arr.shape != py_arr.shape:
                print(
                    f"[ERROR] Shape mismatch for {base}: cpp {cpp_arr.shape} vs python {py_arr.shape}",
                    file=sys.stderr,
                )
                success_this_suffix = False
                overall_success = False
                continue

            if not np.allclose(cpp_arr, py_arr, atol=atol, rtol=rtol):
                diff = np.abs(cpp_arr - py_arr)
                max_abs = diff.max()
                print(
                    f"[ERROR] Value mismatch for {base}: max abs diff {max_abs:.3e} "
                    f"(atol={atol}, rtol={rtol})",
                    file=sys.stderr,
                )
                success_this_suffix = False
                overall_success = False
            else:
                print(f"[OK] {base}: arrays match within tolerances.")

        if success_this_suffix:
            print(f"✓ All comparisons passed for suffix '{suffix}'")
        else:
            print(f"✗ Some comparisons failed for suffix '{suffix}'")

    return overall_success


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Physics benchmark data helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "generate", help="Generate random physics events and save .npy inputs."
    )
    gen.add_argument(
        "--num-events",
        type=int,
        default=50_000,
        help="Number of events to generate (default: 50k).",
    )
    gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    gen.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT,
        help=f"Directory to store .npy files (default: {REPO_ROOT}).",
    )

    cmp_parser = subparsers.add_parser(
        "compare", help="Compare C++ and Python invariant-mass outputs."
    )
    cmp_parser.add_argument(
        "--cpp-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing masses_*_cpp.npy files (default: repo root).",
    )
    cmp_parser.add_argument(
        "--python-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing masses_*_python.npy files (default: repo root).",
    )
    cmp_parser.add_argument(
        "--atol", type=float, default=2e-3, help="Absolute tolerance for comparison."
    )
    cmp_parser.add_argument(
        "--rtol", type=float, default=5e-3, help="Relative tolerance for comparison."
    )
    cmp_parser.add_argument(
        "--suffixes",
        type=str,
        nargs="+",
        default=["_4", "_5", "_6", "_7"],
        help="List of suffixes to compare (default: _4 _5 _6 _7). Use empty string for no suffix.",
    )

    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate 4 sets of inputs with different sizes
        exponents = [4, 5, 6, 7]
        for exp in exponents:
            num_events = 10**exp
            suffix = f"_{exp}"
            print(f"\n{'=' * 60}")
            print(f"Generating dataset with 10^{exp} = {num_events:,} events")
            print(f"{'=' * 60}")
            data = generate_random_events(num_events, args.seed)
            save_generated_data(data, output_dir, suffix)
        return 0

    if args.command == "compare":
        ok = compare_outputs(
            args.cpp_dir.resolve(),
            args.python_dir.resolve(),
            args.atol,
            args.rtol,
            args.suffixes,
        )
        return 0 if ok else 1

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
