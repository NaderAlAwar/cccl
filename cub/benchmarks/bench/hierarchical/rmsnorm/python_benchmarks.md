# Python RMSNorm Benchmarks

These benchmarks use `cuda.bench`, the Python bindings for NVBench.

They are Python-side baselines colocated with the C++ RMSNorm benchmarks:

- `pytorch_rmsnorm.py`: PyTorch `torch.nn.functional.rms_norm`
- `flashinfer_cutedsl_rmsnorm.py`: direct FlashInfer CuTe DSL RMSNorm
- `cutile_rmsnorm.py`: CuTile RMSNorm variants

Example:

```bash
python cub/benchmarks/bench/hierarchical/rmsnorm/pytorch_rmsnorm.py \
  --devices 0 \
  --stopping-criterion entropy \
  --json pytorch_rmsnorm.json
```

All scripts expose the same core axes:

- `T{ct}`: `F32`, `F16`, `BF16`
- `BatchSize`: `64`, `800`, `150000`
- `HiddenSize`: the FlashInfer/CuTe hidden-size sweep used by the C++ benchmarks
- `ZeroData`: `0`, `1`

Optional dependencies are imported lazily. Missing PyTorch, FlashInfer CuTe DSL,
or CuTile support causes the affected benchmark rows to be skipped.
