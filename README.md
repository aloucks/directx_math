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
| Load/Store/Accessors | :heavy_check_mark: * |
| Plane | :heavy_check_mark: |
| Collision | :x: |
| Documentation | :heavy_check_mark: :construction: ** |

* Some alternate forms of the load/store operations are not yet implemented.

** All functions have links to the [DirectXMath documentation].

## SIMD intrinsics

| x86 / x86-64 | ARM | No intrinsics fallback |
| ------------ | --- | ---------------------- |
| :heavy_check_mark: | :x: (will fallback to no-intinsics) | :heavy_check_mark: |

`sse3`, `sse4.1`, `avx`, `fma`, and `avx2` instructions are be enabled automatically when the CPU
target feature is configured.

### Example

    RUSTFLAGS="-C target-feature=+sse3,+sse4.1,+avx,+fma,+avx2" cargo build

Intrinsics may be disabled with the `no_intrinsics` feature. This is the default if SIMD intrinsics
are not available on the target architecture. 

## License

* [DirectXMath] is Copyright (c) Microsoft Corporation.
* DirectXMath for Rust is an independent translation and not affiliated with Microsoft Corporation.
* MIT license ([LICENSE-MIT] or http://opensource.org/licenses/MIT)


[LICENSE-MIT]: LICENSE-MIT
[DirectXMath]: https://github.com/microsoft/DirectXMath
[DirectXMath documentation]: https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference

