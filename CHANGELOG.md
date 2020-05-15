# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
- Documentation updates
- Aligned load and store for vector3/4 and matrix
- Unsealed Swizzle and Permute traits
## Changed
- PartialEq for XMVector now uses `XMVector4Equal` instead of `XMVector4NearEqual`

## [0.1.0-alpha.4] - 2020-05-10
## Added
- Fixed missing `pub` visibility on `IntersectsSphere` and `IntersectsOrientedBox` methods.
- Documentation updates

## [0.1.0-alpha.3] - 2020-05-09
## Added
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
- XMMatrixInverse now accepts `None` for the determinant, to match the upstream API

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

[Unreleased]: https://github.com/aloucks/directx_math/compare/v0.1.0-alpha.4...HEAD
[0.1.0-alpha.4]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/aloucks/directx_math/releases/tag/v0.1.0-alpha.1
