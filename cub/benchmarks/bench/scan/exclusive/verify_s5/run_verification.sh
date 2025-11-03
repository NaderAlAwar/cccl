#!/bin/bash
# Complete verification workflow for S5 scan implementations
# This script generates test data, runs both implementations, and compares outputs

set -e  # Exit on error

# Default parameters
TIMESTEPS=1024
IS_2D=1
DTYPE="float32"
STATE_DIM=40

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --is-2d)
            IS_2D="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --timesteps N     Number of timesteps (default: 1024)"
            echo "  --is-2d [0|1]     0 for 1D, 1 for 2D (default: 1)"
            echo "  --dtype TYPE      float16, float32, or float64 (default: float32)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "======================================================================="
echo "S5 Scan Verification Workflow"
echo "======================================================================="
echo "Configuration:"
echo "  Timesteps: $TIMESTEPS"
echo "  Mode: $([ $IS_2D -eq 1 ] && echo '2D' || echo '1D')"
echo "  Dtype: $DTYPE"
echo "======================================================================="
echo

# Step 1: Build C++ verification tool if needed
if [ ! -f s5_verify ]; then
    echo "Step 1: Building C++ verification tool..."
    make -f Makefile.verify
    echo "✅ Build complete"
    echo
else
    echo "Step 1: C++ verification tool already built (s5_verify exists)"
    echo
fi

# Step 2: Generate test data
echo "Step 2: Generating test data..."
python3 generate_test_data.py \
    --timesteps $TIMESTEPS \
    --is-2d $IS_2D \
    --dtype $DTYPE \
    --state-dim $STATE_DIM
echo "✅ Test data generated"
echo

# Step 3: Run C++ verification
echo "Step 3: Running C++ verification..."
./s5_verify \
    --timesteps $TIMESTEPS \
    --is-2d $IS_2D \
    --dtype $DTYPE
echo "✅ C++ verification complete"
echo

# Step 4: Run Python verification
echo "Step 4: Running Python verification..."
python3 s5_verify.py \
    --timesteps $TIMESTEPS \
    --is-2d $IS_2D \
    --dtype $DTYPE \
    --state-dim $STATE_DIM
echo "✅ Python verification complete"
echo

# Step 5: Compare outputs
echo "Step 5: Comparing outputs..."
python3 compare_outputs.py --dtype $DTYPE

# Check exit code
if [ $? -eq 0 ]; then
    echo
    echo "======================================================================="
    echo "✅ VERIFICATION SUCCESSFUL: Outputs match!"
    echo "======================================================================="
    exit 0
else
    echo
    echo "======================================================================="
    echo "❌ VERIFICATION FAILED: Outputs do not match!"
    echo "======================================================================="
    exit 1
fi
