#!/bin/bash
# Quick verification test with reasonable defaults
# This is a convenience wrapper around run_verification.sh

echo "Running quick verification test..."
echo "This will test with 256 timesteps, 2D mode, float32 dtype"
echo

./run_verification.sh --timesteps 256 --is-2d 1 --dtype float32
