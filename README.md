[![crates.io](https://img.shields.io/crates/v/directx_math.svg)](https://crates.io/crates/directx_math)
[![docs.rs](https://docs.rs/directx_math/badge.svg)](https://docs.rs/directx_math)
[![tests](https://github.com/aloucks/directx_math/workflows/tests/badge.svg)](https://github.com/aloucks/directx_math/actions?query=workflow%3Atests)


# DirectXMath for Rust

A rust translation of [DirectXMath], a SIMD linear algebra library for use in games and graphics apps.

✔️ = Implemented, ❌ = Not yet implemented, 🚧 = In progress

## Implementation status

| API Feature | Implemented |
| --- | ------ |
| Vector | ✔️ |
| Vector 2D | ✔️ |
| Vector 3D | ✔️ |
| Vector 4D | ✔️ |
| Quaternion | ✔️ |
| Matrix | ✔️ |
| Streaming Transforms | ❌ |
| Color | ❌ |
| Load/Store/Accessors | ✔️ &dagger; |
| Plane | ✔️ |
| Collision | ✔️ |
| Documentation | ✔️ &dagger;&dagger; |

&dagger; Some alternate forms of the load/store operations are not yet implemented.

&dagger;&dagger; Most documentation has been ported and all functions have links to the official [DirectXMath documentation].

## SIMD intrinsics

| x86 / x86_64 | ARM | No intrinsics fallback |
| ------------ | --- | ---------------------- |
| ✔️ | ❌ (will fallback to no-intrinsics) | ✔️ |

Intrinsics may be disabled with the `no_intrinsics` feature. This is the implicit behavior
if SIMD intrinsics are not available on the target architecture.

Note that only `sse` and `sse2` are enabled by default for `x86` and `x86-64` targets.
The `sse3`, `sse4.1`, `avx`, `f16c`, `fma`, and `avx2` intrinsics may be enabled via [CPU target features].

### Example: enable all `x86_64` intrinsics

    RUSTFLAGS="-C target-feature=+sse3,+sse4.1,+avx,+f16c,+fma,+avx2" cargo build

## License

* [DirectXMath] is Copyright (c) Microsoft Corporation.
* DirectXMath for Rust is an independent translation and not affiliated with Microsoft Corporation.
* MIT license ([LICENSE-MIT] or http://opensource.org/licenses/MIT)


[LICENSE-MIT]: LICENSE-MIT
[DirectXMath]: https://github.com/microsoft/DirectXMath
[DirectXMath documentation]: https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference
[CPU target features]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute

