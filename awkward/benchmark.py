import sys
from pathlib import Path

import cuda.bench as bench
import cupy as cp
import numpy as np
from playground import physics_analysis_cccl

import awkward as ak

# Add current directory to path to import playground
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("/home/coder/cccl")
PY_OUTPUT_DIR = DATA_DIR


def _load_array(name: str, data_dir: Path, suffix: str = "") -> np.ndarray:
    path = data_dir / f"{name}{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {path}")
    arr = np.load(path)
    # Ensure float arrays are float64
    if arr.dtype == np.float32:
        arr = arr.astype(np.float64)
    return arr


def generate_random_events(data_dir: Path = DATA_DIR, suffix: str = ""):
    """
    Load pre-generated physics events from disk and build the Awkward structure.

    Args:
        data_dir: Directory containing the numpy arrays produced by physics_events.py
        suffix: Suffix to append to file names (e.g., "_4" for 10^4 events)

    Returns:
        Awkward Array with structure matching playground.py events
    """
    electron_pts = _load_array("electron_pts", data_dir, suffix)
    electron_etas = _load_array("electron_etas", data_dir, suffix)
    electron_phis = _load_array("electron_phis", data_dir, suffix)
    muon_pts = _load_array("muons_pts", data_dir, suffix)
    muon_etas = _load_array("muons_etas", data_dir, suffix)
    muon_phis = _load_array("muons_phis", data_dir, suffix)
    electron_offsets = _load_array("electron_offsets", data_dir, suffix)
    muon_offsets = _load_array("muon_offsets", data_dir, suffix)

    if electron_offsets[-1] != electron_pts.size:
        raise ValueError(
            f"electron_offsets total ({electron_offsets[-1]}) does not match electron_pts size ({electron_pts.size})"
        )
    if muon_offsets[-1] != muon_pts.size:
        raise ValueError(
            f"muon_offsets total ({muon_offsets[-1]}) does not match muons_pts size ({muon_pts.size})"
        )

    num_electrons_per_event = np.diff(electron_offsets, prepend=electron_offsets[0])
    num_muons_per_event = np.diff(muon_offsets, prepend=muon_offsets[0])

    electrons = ak.Array(
        {
            "pt": ak.unflatten(electron_pts, num_electrons_per_event),
            "eta": ak.unflatten(electron_etas, num_electrons_per_event),
            "phi": ak.unflatten(electron_phis, num_electrons_per_event),
        }
    )

    muons = ak.Array(
        {
            "pt": ak.unflatten(muon_pts, num_muons_per_event),
            "eta": ak.unflatten(muon_etas, num_muons_per_event),
            "phi": ak.unflatten(muon_phis, num_muons_per_event),
        }
    )

    events = ak.zip({"electrons": electrons, "muons": muons}, depth_limit=1)

    num_events = num_electrons_per_event.size
    print(f"Loaded dataset from {data_dir}")
    print(f"  Events: {num_events:,}")
    print(f"  Total electrons: {electron_pts.size:,}")
    print(f"  Total muons: {muon_pts.size:,}")
    print()

    return events


def _save_python_outputs(result, output_dir: Path, suffix: str = "") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    electron_np = cp.asnumpy(result["electron"])
    muon_np = cp.asnumpy(result["muon"])
    np.save(output_dir / f"masses_electrons{suffix}_python.npy", electron_np)
    np.save(output_dir / f"masses_muons{suffix}_python.npy", muon_np)
    print(f"Saved Python outputs to {output_dir} with suffix '{suffix}'")
    return {"electron": electron_np, "muon": muon_np}


def benchmark_analysis(state):
    """
    Benchmark the three analysis approaches with warmup runs.
    Warmup runs are excluded from timing (only measure steady-state performance).

    Args:
        state: Benchmark state containing parameters like 10Power
    """
    # Get the power of 10 parameter and construct the suffix
    power_of_10 = int(state.get_int64("10Power{io}"))
    suffix = f"_{power_of_10}"

    print(f"Running benchmark for 10^{power_of_10} events (suffix: {suffix})")

    # Load events from the corresponding dataset
    events = generate_random_events(data_dir=DATA_DIR, suffix=suffix)

    events_gpu = ak.to_backend(events, "cuda")
    result_cccl = physics_analysis_cccl(events_gpu)
    _save_python_outputs(result_cccl, PY_OUTPUT_DIR, suffix=suffix)

    cp.cuda.Device().synchronize()

    def launcher(launch):
        physics_analysis_cccl(events_gpu)
        cp.cuda.Device().synchronize()

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    copied_bench = bench.register(benchmark_analysis)
    copied_bench.add_int64_axis("10Power{io}", list(range(4, 8, 1)))
    bench.run_all_benchmarks(sys.argv)
