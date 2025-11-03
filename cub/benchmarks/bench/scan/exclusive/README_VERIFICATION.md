# S5 Scan Verification

To verify that the C++ and Python implementations of the S5 associative scan produce identical results, use the verification tools in the `verify_s5/` directory.

## Quick Start

```bash
cd verify_s5
./run_verification.sh --timesteps 1024 --is-2d 1 --dtype float32
```

See `verify_s5/VERIFICATION_README.md` for complete documentation.
