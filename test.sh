#!/bin/bash

set -x -e

cargo test
cargo test --features no_intrinsics

RUSTFLAGS="-C target-cpu=native" cargo test
RUSTFLAGS="-C target-cpu=native" cargo test --features no_avx2
RUSTFLAGS="-C target-cpu=native" cargo test --features no_fma3
RUSTFLAGS="-C target-cpu=native" cargo test --features no_sse4
RUSTFLAGS="-C target-cpu=native" cargo test --features no_sse3
RUSTFLAGS="-C target-cpu=native" cargo test --features no_intrinsics

# RUSTFLAGS="-C target-feature=+avx2,sse4.1" cargo test

