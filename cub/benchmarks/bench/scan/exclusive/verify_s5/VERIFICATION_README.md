# S5 Scan Verification Guide

This directory (`verify_s5/`) contains tools to verify that the C++ (CUDA/CUB) and Python (PyTorch) implementations of the S5 associative scan produce identical results.

## Overview

The verification workflow consists of:

1. **Data Generation**: Create identical input data saved as `.npy` files
2. **C++ Verification**: Run the CUDA implementation and save outputs
3. **Python Verification**: Run the PyTorch implementation and save outputs
4. **Comparison**: Compare the outputs with appropriate tolerances

## Quick Start

Navigate to the verification directory and run the automated verification workflow:

```bash
cd cub/benchmarks/bench/scan/exclusive/verify_s5
./run_verification.sh --timesteps 1024 --is-2d 1 --dtype float32
```

This will:
- Generate test data
- Build the C++ verification tool (if needed)
- Run both implementations
- Compare the outputs
- Report success or failure

## Files

### Core Files (in parent directory)
- `../s5_operator.cu` - Original CUB benchmark
- `../s5_operator.py` - Original PyTorch benchmark

### Verification Directory Files
- `libnpy/` - Library for reading/writing `.npy` files in C++

### Verification Tools
- `generate_test_data.py` - Generate input data as `.npy` files
- `s5_verify.cu` - C++ verification program (reads/writes `.npy`)
- `s5_verify.py` - Python verification program (reads/writes `.npy`)
- `compare_outputs.py` - Compare C++ and Python outputs
- `run_verification.sh` - Automated workflow script
- `Makefile.verify` - Build system for C++ verification tool

## Manual Usage

### Step 1: Generate Test Data

```bash
python3 generate_test_data.py \
    --timesteps 1024 \
    --is-2d 1 \
    --dtype float32 \
    --state-dim 40
```

This creates:
- `A_in.npy` - Input tensor A
- `Bu_in.npy` - Input tensor Bu

### Step 2: Build C++ Verification Tool

```bash
make -f Makefile.verify
```

This creates the `s5_verify` executable.

**Note**: You may need to adjust `CUDA_ARCH` in the Makefile to match your GPU architecture:
- RTX 30XX: `-arch=sm_86`
- A100: `-arch=sm_80`
- RTX 40XX: `-arch=sm_89`

### Step 3: Run C++ Verification

```bash
./s5_verify --timesteps 1024 --is-2d 1 --dtype float32
```

This creates:
- `A_out_cpp.npy` - C++ output tensor A
- `Bu_out_cpp.npy` - C++ output tensor Bu

### Step 4: Run Python Verification

```bash
python3 s5_verify.py --timesteps 1024 --is-2d 1 --dtype float32
```

This creates:
- `A_out_python.npy` - Python output tensor A
- `Bu_out_python.npy` - Python output tensor Bu

### Step 5: Compare Outputs

```bash
python3 compare_outputs.py --dtype float32
```

This will:
- Load all four output files
- Compare C++ vs Python outputs
- Report differences and success/failure

## Parameters

### --timesteps N
Number of timesteps (sequence length). Common values:
- Small: 256, 512
- Medium: 1024, 2048
- Large: 4096, 8192

### --is-2d [0|1]
- `0`: 1D mode (scalar elements)
- `1`: 2D mode (vector elements with elementwise operations)

### --dtype TYPE
Data type for the scan:
- `float16`: Half precision (FP16)
- `float32`: Single precision (FP32, default)
- `float64`: Double precision (FP64)

**Tolerance defaults**:
- `float16`: rtol=1e-3, atol=1e-5
- `float32`: rtol=1e-4, atol=1e-6 (updated for scan accumulation errors)
- `float64`: rtol=1e-7, atol=1e-10

### --state-dim N
Hidden dimension size (only for 2D mode). Default: 40

## Examples

### Verify 1D scan with single precision

```bash
./run_verification.sh --timesteps 512 --is-2d 0 --dtype float32
```

### Verify 2D scan with half precision

```bash
./run_verification.sh --timesteps 2048 --is-2d 1 --dtype float16
```

### Verify large 2D scan with double precision

```bash
./run_verification.sh --timesteps 8192 --is-2d 1 --dtype float64
```

## Understanding the S5 Operator

The S5 (Structured State Space Sequence) operator is an associative scan operation:

```
(A_i, Bu_i) ⊕ (A_j, Bu_j) = (A_j * A_i, A_j * Bu_i + Bu_j)
```

**1D Mode**: Operates on scalar elements
- Input: Two vectors of scalars (A, Bu)
- Output: Two vectors of scalars

**2D Mode**: Operates on vector elements with elementwise operations
- Input: Two matrices (timesteps × state_dim)
- Output: Two matrices (timesteps × state_dim)
- Each row is treated as a vector element
- Operations are elementwise within each row

## Troubleshooting

### libnpy not found

If compilation fails with "libnpy/include/npy.hpp not found":

```bash
# Make sure libnpy is cloned
git clone https://github.com/llohse/libnpy.git
```

### CUDA architecture mismatch

If you get PTX/SASS errors, update `CUDA_ARCH` in `Makefile.verify`:

```bash
# Check your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv
```

### PyTorch not found

Ensure PyTorch with CUDA support is installed:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Outputs don't match

1. Check for NaN/Inf values in inputs or outputs
2. Try with higher precision dtype (float64)
3. Reduce timesteps to narrow down the issue
4. Examine the specific location of maximum difference reported by `compare_outputs.py`

### float16 comparison fails

FP16 has limited precision. The default tolerance (rtol=1e-3) should handle most cases, but you can adjust:

```bash
python3 compare_outputs.py --dtype float16 --rtol 1e-2 --atol 1e-4
```

## Integration with Benchmarks

The verification tools are separate from the benchmarks (`s5_operator.cu` and `s5_operator.py`) and can be run independently. They share the same operator implementations to ensure consistency.

To verify benchmark results:
1. Run the benchmarks normally
2. Run the verification workflow with the same parameters
3. If verification passes, the benchmark results are trustworthy

## Technical Details

### Why libnpy?

The `libnpy` library provides a simple C++ interface for reading/writing NumPy's `.npy` file format, enabling easy data exchange between C++ and Python without implementing custom binary formats.

### Data Layout

- **1D data**: Stored as flat arrays of length `timesteps`
- **2D data**: Stored as row-major matrices of shape `(timesteps, state_dim)`

Both C++ and Python use the same data layout for compatibility.

### Comparison Tolerances

Floating-point operations have inherent precision limits. The comparison tool uses:
- **Relative tolerance (rtol)**: Relative difference allowed between values
- **Absolute tolerance (atol)**: Absolute difference allowed for small values

Values are considered equal if: `|a - b| <= atol + rtol * |b|`

## Contributing

When modifying the scan implementations:

1. Update both `s5_operator.cu` and `s5_operator.py`
2. Update verification tools (`s5_verify.cu` and `s5_verify.py`) if interface changes
3. Run verification to ensure implementations still match
4. Update this README if new features are added
