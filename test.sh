#!/bin/bash

set -x -e

# cargo test
# cargo test --features no_intrinsics

# RUSTFLAGS="-C target-cpu=native" cargo test
# RUSTFLAGS="-C target-cpu=native" cargo test --features no_avx2
# RUSTFLAGS="-C target-cpu=native" cargo test --features no_avx
# RUSTFLAGS="-C target-cpu=native" cargo test --features no_fma3
# RUSTFLAGS="-C target-cpu=native" cargo test --features no_sse4
# RUSTFLAGS="-C target-cpu=native" cargo test --features no_sse3
# RUSTFLAGS="-C target-cpu=native" cargo test --features no_intrinsics

# RUSTFLAGS="-C target-feature=+avx2,sse4.1" cargo test

RUSTFLAGS="-C target-cpu=x86-64" cargo test
RUSTFLAGS="-C target-cpu=x86-64" cargo test --features="no_intrinsics"
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=+sse3" cargo test
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=+sse3,+sse4.1" cargo test
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=+sse3,+sse4.1,+avx" cargo test
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=+sse3,+sse4.1,+avx,+fma" cargo test
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=+sse3,+sse4.1,+avx,+f16c,+fma" cargo test
RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=+sse3,+sse4.1,+avx,+f16c,+fma,+avx2" cargo test
RUSTFLAGS="-C target-cpu=x86-64" cargo +nightly test --features="specialization"
RUSTFLAGS="-C target-cpu=x86-64" cargo +nightly test --features="specialization,no_intrinsics"
cargo +nightly build --target aarch64-pc-windows-msvc
cargo +nightly build --target aarch64-pc-windows-msvc --features="no_intrinsics"
cargo +nightly build --target aarch64-pc-windows-msvc --no-default-features
