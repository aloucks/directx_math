[![crates.io](https://img.shields.io/crates/v/directx_math.svg)](https://crates.io/crates/directx_math)
[![docs.rs](https://docs.rs/directx_math/badge.svg)](https://docs.rs/directx_math)
[![tests](https://github.com/aloucks/directx_math/workflows/tests/badge.svg)](https://github.com/aloucks/directx_math/actions?query=workflow%3Atests)


# DirectXMath for Rust

A pure rust translation of [DirectXMath], an all inline SIMD linear algebra library for use in
games and graphics apps.

:heavy_check_mark: = Implemented, :x: = Not yet implemented, :construction: = In progress

## Implementation status

| API | Implemented |
| --- | ------ |
| Vector | :heavy_check_mark: |
| Vector 2D | :heavy_check_mark: |
| Vector 3D | :heavy_check_mark: |
| Vector 4D | :heavy_check_mark: |
| Quaternion | :heavy_check_mark: |
| Matrix | :heavy_check_mark: |
| Streaming Transforms | :x: |
| Color | :x: |
| Load/Store/Accessors | :heavy_check_mark: &dagger; |
| Plane | :heavy_check_mark: |
| Collision | :x: |
| Documentation | :heavy_check_mark: :construction: &dagger;&dagger; |

&dagger; Some alternate forms of the load/store operations are not yet implemented.

&dagger;&dagger; All functions have links to the [DirectXMath documentation].

## SIMD intrinsics

| x86 / x86-64 | ARM | No intrinsics fallback |
| ------------ | --- | ---------------------- |
| :heavy_check_mark: | :x: (will fallback to no-intrinsics) | :heavy_check_mark: |

Intrinsics may be disabled with the `no_intrinsics` feature. This is the implicit behavior
if SIMD intrinsics are not available on the target architecture.

Note that only `sse` and `sse2` are enabled by default for `x86` and `x86-64` targets.
The `sse3`, `sse4.1`, `avx`, `fma`, and `avx2` intrinsics may be enabled via [CPU target features].

### Example: enable all `x86/64` intrinsics

    RUSTFLAGS="-C target-feature=+sse3,+sse4.1,+avx,+fma,+avx2" cargo build

## License

* [DirectXMath] is Copyright (c) Microsoft Corporation.
* DirectXMath for Rust is an independent translation and not affiliated with Microsoft Corporation.
* MIT license ([LICENSE-MIT] or http://opensource.org/licenses/MIT)


[LICENSE-MIT]: LICENSE-MIT
[DirectXMath]: https://github.com/microsoft/DirectXMath
[DirectXMath documentation]: https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference
[CPU target features]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute

