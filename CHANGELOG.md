# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Enable `!XMPlaneIsInfinite` debug assertions in `XMMatrixReflect` and `XMMatrixShadow`.
- Fixed clippy warning with `u32` to `f32` conversion.
- All collision functions are now marked with `#[inline]`

## [0.2.0] - 2020-09-05
### Fixed
- Fixed compilation for ARM targets
### Added
- `LoadFloat3x4`, `StoreFloat3x4`, `LoadFloat4x3`, `StoreFloat4x3`
- `XMConvertFloatToHalf`, `XMConvertHalfToFloat`
### Changed
- `BoundingFrustum::CreateFromMatrix` now has a flag to allow right-handed projection matrices.

## [0.1.2] - 2020-06-08
### Changed
- Moved benchmarks into a separate crate

## [0.1.1] - 2020-06-07
### Added
- Added clippy to CI
- Documentation updates. Added additional info from the older DirectX 9 docs.
- `XMFLOAT3X3` From and Into `[f32; 9]`
### Fixed
- Fixed the value for `XM_CRMASK_CR6`
- Fixed `Into<[[f32; 3]; 3]>` for `XMFLOAT3X3`

## [0.1.0] - 2020-05-26
### Added
- Documentation updates
### Changed
- Fixed `XMStoreFloat3A` to correctly accept `&mut Align16<XMFLOAT3>` instead of `&mut XMFLOAT3`.

## [0.1.0-alpha.6] - 2020-05-22
### Added
- Documentation updates
### Changed
- `CreateFromPoints` for `BoundingSphere`, `BoundingBox`, and `BoundingOrientedBox` now
  take `pPoints` as an `Interator` instead of `IntoIterator`. This is due to the `Clone`
  restriction. It's currently difficult or impossible to `map` the items and produce a
  result that both implments `IntoIterator` and `Clone`.

## [0.1.0-alpha.5] - 2020-05-18
## Added
- Documentation updates
- Aligned load and store for vector3/4 and matrix
- Unsealed `Swizzle` and `Permute` traits
### Changed
- `PartialEq` for `XMVector` now uses `XMVector4Equal` instead of `XMVector4NearEqual`

## [0.1.0-alpha.4] - 2020-05-10
### Added
- Fixed missing `pub` visibility on `IntersectsSphere` and `IntersectsOrientedBox` methods.
- Documentation updates

## [0.1.0-alpha.3] - 2020-05-09
### Added
- Documentation updates
### Changed
- Removed all traits from collision API. Duplicate method names have been updated to
  have suffixes that match their parameter types. For example,
  `BoundingBox::Contains(&BoundingSphere)` and `BoundingBox::Contains(&BoundingBox)` are now
  `BoundingBox::ContainsSphere(&BoundingSphere)` and `BoundingBox::ContainsBox(&BoundingBox)`.
- Removed `Ray`, `Direction`, `Triangle`, and related helper typedefs now that the
  `Contains` and `Intersects` traits no longer exist.

## [0.1.0-alpha.2] - 2020-05-07
### Added
- Documentation updates
- Collision API
### Changed
- `XMMatrixInverse` now accepts `None` for the determinant, to match the upstream API

## [0.1.0-alpha.1] - 2020-05-02
### Added
- Vector
- 2D Vector
- 3D Vector
- 4D Vector
- Quaternion
- Matrix
- Load/Store/Accessors
- Plane

[Unreleased]: https://github.com/aloucks/directx_math/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/aloucks/directx_math/releases/tag/v0.2.0
[0.1.2]: https://github.com/aloucks/directx_math/releases/tag/v0.1.2
[0.1.1]: https://github.com/aloucks/directx_math/releases/tag/v0.1.1
[0.1.0]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0
[0.1.0-alpha.6]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.6
[0.1.0-alpha.5]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.5
[0.1.0-alpha.4]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.1
