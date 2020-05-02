//! # DirectX Math for Rust
//!
//! A pure rust translation of [DirectXMath], an all inline SIMD linear algebra library for use in
//! games and graphics apps.
//!
//! All functions and structs are exported at the crate root. Modules are organized according
//! to the `C++` [reference documentation].
//!
//! [DirectXMath]: https://github.com/microsoft/DirectXMath
//! [reference documentation]: https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions
//!
//! ## Row-major matrices
//!
//! DirectXMath uses **row-major** matrices, row vectors, and pre-multiplication. Handedness is determined by
//! which function version is used (RH vs. LH).
//!
//! **Row-major** multiplication order:
//!
//! ```text
//! MVP = Model * View * Projection;
//! ```
//!
//! **Column-major** multiplication order:
//!
//! ```text
//! MVP = Projection * View * Model;
//! ```
//!
//! ## Example
//! ```rust
//! use directx_math::*;
//!
//! let eye = XMVectorSet(10.0, 10.0, 10.0, 0.0);
//! let focus = XMVectorSet(0.0, 0.0, 0.0, 0.0);
//! let up = XMVectorSet(0.0, 1.0, 0.0, 0.0);
//! let view = XMMatrixLookAtRH(eye, focus, up);
//!
//! let fov_y = XMConvertToRadians(65.0);
//! let aspect_ratio = 1024.0 / 768.0;
//! let near_z = 0.1;
//! let far_z = 1000.0;
//! let projection = XMMatrixPerspectiveFovRH(fov_y, aspect_ratio, near_z, far_z);
//!
//! let model = XMMatrixIdentity();
//!
//! let mvp = XMMatrixMultiply(XMMatrixMultiply(model, &view), &projection);
//!
//! // or use the unit struct with operator overloads
//! let model = XMMatrix(model);
//! let view = XMMatrix(view);
//! let projection = XMMatrix(projection);
//!
//! // row-major multiplication order is the same as the transformation order
//! assert_eq!(XMMatrix(mvp), model * view * projection);
//! ```


// ## Conversion Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-conversion>
//
// ## Matrix Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-matrix>
//
// ## Plane Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-plane>
//
// ## Quaternion Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-quaternion>
//
// ## Scalar Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-scalar>
//
// ## Vector Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector>
//
// ## 2D Vector Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector2>
//
// ## 3D Vector Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector3>
//
// ## 4D Vector Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector4>
//
// ## Template Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-templates>
//
// ## Triangle Test Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-triangletests>
//
// ## Utility Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-utilities>
//
// ## Vector Accessor Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-accessors>
//
// ## Vector Load Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-load>
//
// ## Vector Store Functions
//
// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-store>

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(unused_parens)]

// TODO: Change allow unused_macros to deny
#![allow(unused_macros)]

#![deny(unreachable_code)]
#![deny(unused_variables)]
#![deny(unused_unsafe)]
#![deny(dead_code)]
#![deny(unused_mut)]
#![deny(unused_assignments)]

#![cfg_attr(nightly_specialization, feature(min_specialization))]
#![cfg_attr(nightly_specialization, feature(specialization))]

#[allow(unused_imports)]
use std::mem;

#[cfg(all(target_arch="x86_64", not(_XM_NO_INTRINSICS_)))]
#[doc(hidden)]
use std::arch::x86_64 as arch;

#[cfg(all(target_arch="x86", not(_XM_NO_INTRINSICS_)))]
#[doc(hidden)]
use std::arch::x86 as arch;

#[cfg(all(target_arch="arm", not(_XM_NO_INTRINSICS_)))]
use std::arch::arm as arch;

#[cfg(not(_XM_NO_INTRINSICS_))]
use arch::*;

/// A utility function for creating masks to use with Intel shuffle and permute intrinsics.
///
/// Note this is the exact implementation of `core::arch::x86_64::_MM_SHUFFLE` which
/// is nightly-only.
#[inline(always)]
#[allow(dead_code)]
pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn uninitialized<T>() -> T {
    mem::MaybeUninit::uninit().assume_init()
}

#[inline(always)]
pub(crate) fn fabsf(x: f32) -> f32 {
    x.abs()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn floorf(x: f32) -> f32 {
    x.floor()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn ceilf(x: f32) -> f32 {
    x.ceil()
}

#[inline(always)]
pub(crate) fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn sinf(x: f32) -> f32 {
    x.sin()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn cosf(x: f32) -> f32 {
    x.cos()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn tanf(x: f32) -> f32 {
    x.tan()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn asinf(x: f32) -> f32 {
    x.asin()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn acosf(x: f32) -> f32 {
    x.acos()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn atanf(x: f32) -> f32 {
    x.atan()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn atan2f(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

#[inline(always)]
pub(crate) fn powf(x: f32, n: f32) -> f32 {
    x.powf(n)
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn sinh(x: f32) -> f32 {
    x.sinh()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn cosh(x: f32) -> f32 {
    x.cosh()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn tanh(x: f32) -> f32 {
    x.tanh()
}

#[inline(always)]
#[cfg(_XM_NO_INTRINSICS_)]
#[allow(dead_code)]
// TODO: XMVectorLog2
pub(crate) fn logf(x: f32) -> f32 {
    x.log2()
}

#[inline]
#[cfg(_XM_NO_INTRINSICS_)]
pub(crate) fn modff(x: f32) -> (f32, f32) {
    // https://github.com/rust-lang/libm/blob/d3f1dba56dc47fbae6f5d6e47c3a00a4aab5b6c5/src/math/modff.rs#L1
    // https://github.com/rust-lang/libm/blob/d3f1dba56dc47fbae6f5d6e47c3a00a4aab5b6c5/LICENSE-MIT
    let rv2: f32;
    let mut u: u32 = x.to_bits();
    let mask: u32;
    let e = ((u >> 23 & 0xff) as i32) - 0x7f;

    /* no fractional part */
    if e >= 23 {
        rv2 = x;
        if e == 0x80 && (u << 9) != 0 {
            /* nan */
            return (x, rv2);
        }
        u &= 0x80000000;
        return (f32::from_bits(u), rv2);
    }
    /* no integral part */
    if e < 0 {
        u &= 0x80000000;
        rv2 = f32::from_bits(u);
        return (x, rv2);
    }

    mask = 0x007fffff >> e;
    if (u & mask) == 0 {
        rv2 = x;
        u &= 0x80000000;
        return (f32::from_bits(u), rv2);
    }
    u &= !mask;
    rv2 = f32::from_bits(u);
    return (x - rv2, rv2);
}

#[inline(always)]
#[allow(dead_code)]
const fn ubool(a: u32) -> bool {
    a != 0
}

#[inline(always)]
#[allow(dead_code)]
const fn ibool(a: i32) -> bool {
    a != 0
}

#[repr(C, align(16))]
struct Align16<T>(T);

impl<T> std::ops::Deref for Align16<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Align16<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! idx {
    (mut $name:ident[$offset:expr]) => {
        *$name.add($offset as usize)
    };
    ($name:ident[$offset:expr]) => {
        *$name.offset($offset as isize)
    };
    ($name:ident[$offset:expr][$offset2:expr]) => {
        *$name[$offset as usize].offset($offset2 as isize)
    };
    (f32x4($m128:expr)[$offset:expr]) => {{
        let vector: &[f32; 4] = &std::mem::transmute($m128);
        vector[$offset]
    }};
}

macro_rules! assert_approx_eq  {
    ($a:expr, $b:expr) => {{
        let eps = 1.0e-6;
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (*a - *b).abs()
        );
    }};
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        let eps = $eps;
        assert!(
            (*a - *b).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            eps,
            (*a - *b).abs()
        );
    }};
}

// --

#[cfg(_XM_SSE_INTRINSICS_)]
macro_rules! XM_STREAM_PS {
    ($p:expr, $a:expr) => {
        $crate::arch::_mm_stream_ps(($p), ($a))
    }
}

#[cfg(_XM_SSE_INTRINSICS_)]
macro_rules! XM256_STREAM_PS {
    ($p:expr, $a:expr) => {
        $crate::arch::_mm256_stream_ps(($p), ($a))
    }
}

#[cfg(_XM_SSE_INTRINSICS_)]
macro_rules! XM_SFENCE {
    () => {
        $crate::arch::_mm_sfence()
    }
}

// --

#[cfg(_XM_FMA3_INTRINSICS_)]
macro_rules! XM_FMADD_PS {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::arch::_mm_fmadd_ps(($a), ($b), ($c))
    }
}

#[cfg(_XM_FMA3_INTRINSICS_)]
macro_rules! XM_FNMADD_PS {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::arch::_mm_fnmadd_ps(($a), ($b), ($c))
    }
}

// --

#[cfg(all(not(_XM_FMA3_INTRINSICS_), _XM_SSE_INTRINSICS_))]
macro_rules! XM_FMADD_PS {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::arch::_mm_add_ps(_mm_mul_ps(($a), ($b)), ($c))
    }
}

#[cfg(all(not(_XM_FMA3_INTRINSICS_), _XM_SSE_INTRINSICS_))]
macro_rules! XM_FNMADD_PS {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::arch::_mm_sub_ps(($c), _mm_mul_ps(($a), ($b)))
    }
}

// --

#[cfg(all(_XM_AVX_INTRINSICS_, _XM_FAVOR_INTEL_))]
macro_rules! XM_PERMUTE_PS {
    ($v:expr, $c:expr) => {
        $crate::arch::_mm_permute_ps(($v), $c)
    }
}

#[cfg(all(not(all(_XM_AVX_INTRINSICS_, _XM_FAVOR_INTEL_)), _XM_SSE_INTRINSICS_))]
macro_rules! XM_PERMUTE_PS {
    ($v:expr, $c:expr) => {
        $crate::arch::_mm_shuffle_ps(($v), ($v), $c)
    }
}

// --

macro_rules! XM_PREFETCH {
    ($a:expr) => {
        // TODO: builtin
    }
}

mod vector;
mod convert;
mod globals;
mod misc;
mod matrix;

pub use vector::*;
pub use convert::*;
use globals::*;
pub use misc::*;
pub use matrix::*;

mod doc {
    // /// Color functions
    // ///
    // /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-color>
    // pub mod color {
    //     // TODO: pub use crate::misc::XMColorSRGBToRGB;
    //     // TODO: pub use crate::misc::XMColorRGBToSRGB;
    //     // TODO: pub use crate::misc::XMColorAdjustContrast;
    //     // TODO: pub use crate::misc::XMColorAdjustSaturation;
    //     // TODO: pub use crate::misc::XMColorEqual;
    //     // TODO: pub use crate::misc::XMColorGreater;
    //     // TODO: pub use crate::misc::XMColorGreaterOrEqual;
    //     // TODO: pub use crate::misc::XMColorHSLToRGB;
    //     // TODO: pub use crate::misc::XMColorHSVToRGB;
    //     // TODO: pub use crate::misc::XMColorIsInfinite;
    //     // TODO: pub use crate::misc::XMColorIsNaN;
    //     // TODO: pub use crate::misc::XMColorLess;
    //     // TODO: pub use crate::misc::XMColorLessOrEqual;
    //     // TODO: pub use crate::misc::XMColorModulate;
    //     // TODO: pub use crate::misc::XMColorNegative;
    //     // TODO: pub use crate::misc::XMColorNotEqual;
    //     // TODO: pub use crate::misc::XMColorRGBToHSL;
    //     // TODO: pub use crate::misc::XMColorRGBToHSV;
    //     // TODO: pub use crate::misc::XMColorRGBToXYZ;
    //     // TODO: pub use crate::misc::XMColorRGBToYUV;
    //     // TODO: pub use crate::misc::XMColorRGBToYUV_HD;
    //     // TODO: pub use crate::misc::XMColorSRGBToXYZ;
    //     // TODO: pub use crate::misc::XMColorXYZToRGB;
    //     // TODO: pub use crate::misc::XMColorXYZToSRGB;
    //     // TODO: pub use crate::misc::XMColorYUVToRGB;
    //     // TODO: pub use crate::misc::XMColorYUVToRGB_HD;
    // }

    /// Vector data conversion functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-conversion>
    pub mod conversion {
        // TODO: pub use crate::convert::XMConvertFloatToHalf;
        // TODO: pub use crate::convert::XMConvertFloatToHalfStream;
        // TODO: pub use crate::convert::XMConvertHalfToFloat;
        // TODO: pub use crate::convert::XMConvertHalfToFloatStream;
        pub use crate::XMConvertToDegrees;
        pub use crate::XMConvertToRadians;
        pub use crate::convert::XMConvertVectorFloatToInt;
        pub use crate::convert::XMConvertVectorFloatToUInt;
        pub use crate::convert::XMConvertVectorIntToFloat;
        pub use crate::convert::XMConvertVectorUIntToFloat;
    }

    /// Matrix functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-matrix>
    pub mod matrix {
        pub use crate::matrix::XMMatrixAffineTransformation;
        pub use crate::matrix::XMMatrixAffineTransformation2D;
        pub use crate::matrix::XMMatrixDecompose;
        pub use crate::matrix::XMMatrixDeterminant;
        pub use crate::matrix::XMMatrixIdentity;
        pub use crate::matrix::XMMatrixInverse;
        pub use crate::matrix::XMMatrixIsIdentity;
        pub use crate::matrix::XMMatrixIsInfinite;
        pub use crate::matrix::XMMatrixIsNaN;
        pub use crate::matrix::XMMatrixLookAtLH;
        pub use crate::matrix::XMMatrixLookAtRH;
        pub use crate::matrix::XMMatrixLookToLH;
        pub use crate::matrix::XMMatrixLookToRH;
        pub use crate::matrix::XMMatrixMultiply;
        pub use crate::matrix::XMMatrixMultiplyTranspose;
        pub use crate::matrix::XMMatrixOrthographicLH;
        // TODO: pub use crate::matrix::XMMatrixOrthographicOffCenterLH;
        // TODO: pub use crate::matrix::XMMatrixOrthographicOffCenterRH;
        pub use crate::matrix::XMMatrixOrthographicRH;
        pub use crate::matrix::XMMatrixPerspectiveFovLH;
        pub use crate::matrix::XMMatrixPerspectiveFovRH;
        pub use crate::matrix::XMMatrixPerspectiveLH;
        // TODO: pub use crate::matrix::XMMatrixPerspectiveOffCenterLH;
        // TODO: pub use crate::matrix::XMMatrixPerspectiveOffCenterRH;
        pub use crate::matrix::XMMatrixPerspectiveRH;
        pub use crate::matrix::XMMatrixReflect;
        pub use crate::matrix::XMMatrixRotationAxis;
        pub use crate::matrix::XMMatrixRotationNormal;
        pub use crate::matrix::XMMatrixRotationQuaternion;
        pub use crate::matrix::XMMatrixRotationRollPitchYaw;
        pub use crate::matrix::XMMatrixRotationRollPitchYawFromVector;
        pub use crate::matrix::XMMatrixRotationX;
        pub use crate::matrix::XMMatrixRotationY;
        pub use crate::matrix::XMMatrixRotationZ;
        pub use crate::matrix::XMMatrixScaling;
        pub use crate::matrix::XMMatrixScalingFromVector;
        pub use crate::matrix::XMMatrixSet;
        pub use crate::matrix::XMMatrixShadow;
        pub use crate::matrix::XMMatrixTransformation;
        pub use crate::matrix::XMMatrixTransformation2D;
        pub use crate::matrix::XMMatrixTranslation;
        pub use crate::matrix::XMMatrixTranslationFromVector;
        pub use crate::matrix::XMMatrixTranspose;

    }

    /// Plane functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-plane>
    pub mod plane {
        pub use crate::misc::XMPlaneDot;
        pub use crate::misc::XMPlaneDotCoord;
        pub use crate::misc::XMPlaneDotNormal;
        pub use crate::misc::XMPlaneEqual;
        pub use crate::misc::XMPlaneFromPointNormal;
        pub use crate::misc::XMPlaneFromPoints;
        pub use crate::misc::XMPlaneIntersectLine;
        pub use crate::misc::XMPlaneIntersectPlane;
        pub use crate::misc::XMPlaneIsInfinite;
        pub use crate::misc::XMPlaneIsNaN;
        pub use crate::misc::XMPlaneNearEqual;
        pub use crate::misc::XMPlaneNormalize;
        pub use crate::misc::XMPlaneNormalizeEst;
        pub use crate::misc::XMPlaneNotEqual;
        pub use crate::misc::XMPlaneTransform;
        // TODO: pub use crate::misc::XMPlaneTransformStream;
    }

    /// Quaternion functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-quaternion>
    pub mod quaternion {
        pub use crate::misc::XMQuaternionBaryCentric;
        pub use crate::misc::XMQuaternionBaryCentricV;
        pub use crate::misc::XMQuaternionConjugate;
        pub use crate::misc::XMQuaternionDot;
        pub use crate::misc::XMQuaternionEqual;
        pub use crate::misc::XMQuaternionExp;
        pub use crate::misc::XMQuaternionIdentity;
        pub use crate::misc::XMQuaternionInverse;
        pub use crate::misc::XMQuaternionIsIdentity;
        pub use crate::misc::XMQuaternionIsInfinite;
        pub use crate::misc::XMQuaternionIsNaN;
        pub use crate::misc::XMQuaternionLength;
        pub use crate::misc::XMQuaternionLengthSq;
        pub use crate::misc::XMQuaternionLn;
        pub use crate::misc::XMQuaternionMultiply;
        pub use crate::misc::XMQuaternionNormalize;
        pub use crate::misc::XMQuaternionNormalizeEst;
        pub use crate::misc::XMQuaternionNotEqual;
        pub use crate::misc::XMQuaternionReciprocalLength;
        pub use crate::misc::XMQuaternionRotationAxis;
        pub use crate::misc::XMQuaternionRotationMatrix;
        pub use crate::misc::XMQuaternionRotationNormal;
        pub use crate::misc::XMQuaternionRotationRollPitchYaw;
        pub use crate::misc::XMQuaternionRotationRollPitchYawFromVector;
        pub use crate::misc::XMQuaternionSlerp;
        pub use crate::misc::XMQuaternionSlerpV;
        pub use crate::misc::XMQuaternionSquad;
        pub use crate::misc::XMQuaternionSquadSetup;
        pub use crate::misc::XMQuaternionSquadV;
        pub use crate::misc::XMQuaternionToAxisAngle;
    }

    /// Scalar functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-scalar>
    pub mod scalar {
        pub use crate::misc::XMScalarACos;
        pub use crate::misc::XMScalarACosEst;
        pub use crate::misc::XMScalarASin;
        pub use crate::misc::XMScalarASinEst;
        pub use crate::misc::XMScalarCos;
        pub use crate::misc::XMScalarCosEst;
        pub use crate::misc::XMScalarModAngle;
        pub use crate::misc::XMScalarNearEqual;
        pub use crate::misc::XMScalarSin;
        pub use crate::misc::XMScalarSinCos;
        pub use crate::misc::XMScalarSinCosEst;
        pub use crate::misc::XMScalarSinEst;
    }

    /// Vector functions usable on any vector type
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector>
    pub mod vector {
        /// Vector arithmetic functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-arithmetic>
        pub mod arithmetic {
            pub use crate::vector::XMVectorAbs;
            pub use crate::vector::XMVectorAdd;
            pub use crate::vector::XMVectorAddAngles;
            pub use crate::vector::XMVectorCeiling;
            pub use crate::vector::XMVectorClamp;
            pub use crate::vector::XMVectorDivide;
            pub use crate::vector::XMVectorFloor;
            pub use crate::vector::XMVectorIsInfinite;
            pub use crate::vector::XMVectorIsNaN;
            pub use crate::vector::XMVectorMax;
            pub use crate::vector::XMVectorMin;
            pub use crate::vector::XMVectorMod;
            pub use crate::vector::XMVectorModAngles;
            pub use crate::vector::XMVectorMultiply;
            pub use crate::vector::XMVectorMultiplyAdd;
            pub use crate::vector::XMVectorNegate;
            pub use crate::vector::XMVectorNegativeMultiplySubtract;
            pub use crate::vector::XMVectorPow;
            pub use crate::vector::XMVectorReciprocal;
            pub use crate::vector::XMVectorReciprocalEst;
            pub use crate::vector::XMVectorReciprocalSqrt;
            pub use crate::vector::XMVectorReciprocalSqrtEst;
            pub use crate::vector::XMVectorRound;
            pub use crate::vector::XMVectorSaturate;
            pub use crate::vector::XMVectorScale;
            pub use crate::vector::XMVectorSqrt;
            pub use crate::vector::XMVectorSqrtEst;
            pub use crate::vector::XMVectorSubtract;
            pub use crate::vector::XMVectorSubtractAngles;
            pub use crate::vector::XMVectorSum;
            pub use crate::vector::XMVectorTruncate;
        }

        /// Vector bit-wise manipulation functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-bit-wise>
        pub mod bit_wise {
            pub use crate::vector::XMVectorAndCInt;
            pub use crate::vector::XMVectorAndInt;
            pub use crate::vector::XMVectorNorInt;
            pub use crate::vector::XMVectorNotEqual;
            pub use crate::vector::XMVectorNotEqualInt;
            pub use crate::vector::XMVectorOrInt;
            pub use crate::vector::XMVectorXorInt;
        }

        /// Vector comparison functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-comparison>
        pub mod comparison {
            pub use crate::vector::XMVectorEqual;
            pub use crate::vector::XMVectorEqualInt;
            pub use crate::vector::XMVectorEqualIntR;
            pub use crate::vector::XMVectorEqualR;
            pub use crate::vector::XMVectorGreater;
            pub use crate::vector::XMVectorGreaterOrEqual;
            pub use crate::vector::XMVectorGreaterOrEqualR;
            pub use crate::vector::XMVectorGreaterR;
            pub use crate::vector::XMVectorLess;
            pub use crate::vector::XMVectorLessOrEqual;
            pub use crate::vector::XMVectorNearEqual;

        }

        /// Component-wise Vector operations
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-component-wise>
        pub mod component_wise {
            pub use crate::vector::XMVectorInsert;
            pub use crate::vector::XMVectorMergeXY;
            pub use crate::vector::XMVectorMergeZW;
            pub use crate::vector::XMVectorPermute;
            pub use crate::vector::XMVectorRotateLeft;
            pub use crate::vector::XMVectorRotateRight;
            pub use crate::vector::XMVectorSelect;
            pub use crate::vector::XMVectorSelectControl;
            pub use crate::vector::XMVectorShiftLeft;
            pub use crate::vector::XMVectorSplatW;
            pub use crate::vector::XMVectorSplatX;
            pub use crate::vector::XMVectorSplatY;
            pub use crate::vector::XMVectorSplatZ;
            pub use crate::vector::XMVectorSwizzle;

        }

        /// Geometric functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-geometric>
        pub mod geometric {
            pub use crate::vector::XMVectorBaryCentric;
            pub use crate::vector::XMVectorBaryCentricV;
            pub use crate::vector::XMVectorCatmullRom;
            pub use crate::vector::XMVectorCatmullRomV;
            pub use crate::vector::XMVectorHermite;
            pub use crate::vector::XMVectorHermiteV;
            pub use crate::vector::XMVectorInBounds;
            pub use crate::vector::XMVectorInBoundsR;
            pub use crate::vector::XMVectorLerp;
            pub use crate::vector::XMVectorLerpV;            
        }

        /// Vector initialization
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-initialization>
        pub mod initialization {
            pub use crate::vector::XMVectorFalseInt;
            pub use crate::vector::XMVectorReplicate;
            pub use crate::vector::XMVectorReplicateInt;
            // TODO: pub use crate::vector::XMVectorReplicateIntPtr;
            pub use crate::vector::XMVectorReplicatePtr;
            pub use crate::vector::XMVectorSet;
            pub use crate::XMVectorSetBinaryConstant;
            pub use crate::vector::XMVectorSetInt;
            pub use crate::XMVectorSplatConstant;
            pub use crate::XMVectorSplatConstantInt;
            pub use crate::vector::XMVectorSplatEpsilon;
            pub use crate::vector::XMVectorSplatInfinity;
            pub use crate::vector::XMVectorSplatOne;
            pub use crate::vector::XMVectorSplatQNaN;
            pub use crate::vector::XMVectorSplatSignMask;
            pub use crate::vector::XMVectorTrueInt;
            pub use crate::vector::XMVectorZero;
        }

        /// Vector based trigonometry and logarithmic functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector-transcendental>
        pub mod transendental {
            pub use crate::vector::XMVectorACos;
            pub use crate::vector::XMVectorACosEst;
            pub use crate::vector::XMVectorASin;
            pub use crate::vector::XMVectorASinEst;
            pub use crate::vector::XMVectorATan;
            pub use crate::vector::XMVectorATan2;
            pub use crate::vector::XMVectorATan2Est;
            pub use crate::vector::XMVectorATanEst;
            pub use crate::vector::XMVectorCos;
            pub use crate::vector::XMVectorCosEst;
            pub use crate::vector::XMVectorCosH;
            pub use crate::vector::XMVectorExp;
            pub use crate::vector::XMVectorExp2;
            // TODO: pub use crate::vector::XMVectorExpE;
            // TODO: pub use crate::vector::XMVectorLog;
            // TODO: pub use crate::vector::XMVectorLog2;
            // TODO: pub use crate::vector::XMVectorLogE;
            pub use crate::vector::XMVectorSin;
            pub use crate::vector::XMVectorSinCos;
            pub use crate::vector::XMVectorSinCosEst;
            pub use crate::vector::XMVectorSinEst;
            pub use crate::vector::XMVectorSinH;
            pub use crate::vector::XMVectorTan;
            pub use crate::vector::XMVectorTanEst;
            pub use crate::vector::XMVectorTanH;
        }
    }
        /// 2D Vector functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector2>
    pub mod vector2d {
        /// 2D Vector comparison functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector2-comparison>
        pub mod comparison {
            pub use crate::vector::XMVector2Equal;
            pub use crate::vector::XMVector2EqualInt;
            pub use crate::vector::XMVector2EqualIntR;
            pub use crate::vector::XMVector2EqualR;
            pub use crate::vector::XMVector2Greater;
            pub use crate::vector::XMVector2GreaterOrEqual;
            pub use crate::vector::XMVector2GreaterOrEqualR;
            pub use crate::vector::XMVector2GreaterR;
            pub use crate::vector::XMVector2IsInfinite;
            pub use crate::vector::XMVector2IsNaN;
            pub use crate::vector::XMVector2Less;
            pub use crate::vector::XMVector2LessOrEqual;
            pub use crate::vector::XMVector2NearEqual;
            pub use crate::vector::XMVector2NotEqual;
            pub use crate::vector::XMVector2NotEqualInt;
        }

        /// 2D Vector geometric functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector2-geometric>
        pub mod geometric {
            pub use crate::vector::XMVector2AngleBetweenNormals;
            pub use crate::vector::XMVector2AngleBetweenNormalsEst;
            pub use crate::vector::XMVector2AngleBetweenVectors;
            pub use crate::vector::XMVector2ClampLength;
            pub use crate::vector::XMVector2ClampLengthV;
            pub use crate::vector::XMVector2Cross;
            pub use crate::vector::XMVector2Dot;
            pub use crate::vector::XMVector2InBounds;
            pub use crate::vector::XMVector2IntersectLine;
            pub use crate::vector::XMVector2Length;
            pub use crate::vector::XMVector2LengthEst;
            pub use crate::vector::XMVector2LengthSq;
            pub use crate::vector::XMVector2LinePointDistance;
            pub use crate::vector::XMVector2Normalize;
            pub use crate::vector::XMVector2NormalizeEst;
            pub use crate::vector::XMVector2Orthogonal;
            pub use crate::vector::XMVector2ReciprocalLength;
            pub use crate::vector::XMVector2ReciprocalLengthEst;
            pub use crate::vector::XMVector2Reflect;
            pub use crate::vector::XMVector2Refract;
            pub use crate::vector::XMVector2RefractV;
        }

        /// 2D vector transformation
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector2-transformation>
        pub mod transformation {
            pub use crate::vector::XMVector2Transform;
            pub use crate::vector::XMVector2TransformCoord;
            // TODO: pub use crate::vector::XMVector2TransformCoordStream;
            pub use crate::vector::XMVector2TransformNormal;
            // TODO: pub use crate::vector::XMVector2TransformNormalStream;
            // TODO: pub use crate::vector::XMVector2TransformStream;
        }
    }

    /// 3D Vector functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector3>
    pub mod vector3d {
        /// 3D Vector comparison
        ///
        /// https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector3-comparison
        pub mod comparison {
            pub use crate::vector::XMVector3Equal;
            pub use crate::vector::XMVector3EqualInt;
            pub use crate::vector::XMVector3EqualIntR;
            pub use crate::vector::XMVector3EqualR;
            pub use crate::vector::XMVector3Greater;
            pub use crate::vector::XMVector3GreaterOrEqual;
            pub use crate::vector::XMVector3GreaterOrEqualR;
            pub use crate::vector::XMVector3GreaterR;
            pub use crate::vector::XMVector3IsInfinite;
            pub use crate::vector::XMVector3IsNaN;
            pub use crate::vector::XMVector3Less;
            pub use crate::vector::XMVector3LessOrEqual;
            pub use crate::vector::XMVector3NearEqual;
            pub use crate::vector::XMVector3NotEqual;
            pub use crate::vector::XMVector3NotEqualInt;
        }

        /// 3D vector geometric functions
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector3-geometric>
        pub mod geometric {
            pub use crate::vector::XMVector3AngleBetweenNormals;
            pub use crate::vector::XMVector3AngleBetweenNormalsEst;
            pub use crate::vector::XMVector3AngleBetweenVectors;
            pub use crate::vector::XMVector3ClampLength;
            pub use crate::vector::XMVector3ClampLengthV;
            pub use crate::vector::XMVector3ComponentsFromNormal;
            pub use crate::vector::XMVector3Cross;
            pub use crate::vector::XMVector3Dot;
            pub use crate::vector::XMVector3InBounds;
            pub use crate::vector::XMVector3Length;
            pub use crate::vector::XMVector3LengthEst;
            pub use crate::vector::XMVector3LengthSq;
            pub use crate::vector::XMVector3LinePointDistance;
            pub use crate::vector::XMVector3Normalize;
            pub use crate::vector::XMVector3NormalizeEst;
            pub use crate::vector::XMVector3Orthogonal;
            pub use crate::vector::XMVector3ReciprocalLength;
            pub use crate::vector::XMVector3ReciprocalLengthEst;
            pub use crate::vector::XMVector3Reflect;
            pub use crate::vector::XMVector3Refract;
            pub use crate::vector::XMVector3RefractV;
        }

        /// 3D vector transformation
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector3-transformation>
        pub mod transformation {
            pub use crate::vector::XMVector3InverseRotate;
            pub use crate::vector::XMVector3Project;
            // TODO: pub use crate::vector::XMVector3ProjectStream;
            pub use crate::vector::XMVector3Rotate;
            pub use crate::vector::XMVector3Transform;
            pub use crate::vector::XMVector3TransformCoord;
            // TODO: pub use crate::vector::XMVector3TransformCoordStream;
            pub use crate::vector::XMVector3TransformNormal;
            // TODO: pub use crate::vector::XMVector3TransformNormalStream;
            // TODO: pub use crate::vector::XMVector3TransformStream;
            pub use crate::vector::XMVector3Unproject;
            // TODO: pub use crate::vector::XMVector3UnprojectStream;
        }
    }

    /// 4D Vector functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector4>
    pub mod vector4d {
        /// 4D vector comparison
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector4-comparison>
        pub mod comparison {
            pub use crate::vector::XMVector4Equal;
            pub use crate::vector::XMVector4EqualInt;
            pub use crate::vector::XMVector4EqualIntR;
            pub use crate::vector::XMVector4EqualR;
            pub use crate::vector::XMVector4Greater;
            pub use crate::vector::XMVector4GreaterOrEqual;
            pub use crate::vector::XMVector4GreaterOrEqualR;
            pub use crate::vector::XMVector4GreaterR;
            pub use crate::vector::XMVector4IsInfinite;
            pub use crate::vector::XMVector4IsNaN;
            pub use crate::vector::XMVector4Less;
            pub use crate::vector::XMVector4LessOrEqual;
            pub use crate::vector::XMVector4NearEqual;
            pub use crate::vector::XMVector4NotEqual;
            pub use crate::vector::XMVector4NotEqualInt;
        }

        /// 4D vector geometric
        ///
        /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-vector4-geometric>
        pub mod geometric {
            pub use crate::vector::XMVector4AngleBetweenNormals;
            pub use crate::vector::XMVector4AngleBetweenNormalsEst;
            pub use crate::vector::XMVector4AngleBetweenVectors;
            pub use crate::vector::XMVector4ClampLength;
            pub use crate::vector::XMVector4ClampLengthV;
            pub use crate::vector::XMVector4Cross;
            pub use crate::vector::XMVector4Dot;
            pub use crate::vector::XMVector4InBounds;
            pub use crate::vector::XMVector4Length;
            pub use crate::vector::XMVector4LengthEst;
            pub use crate::vector::XMVector4LengthSq;
            pub use crate::vector::XMVector4Normalize;
            pub use crate::vector::XMVector4NormalizeEst;
            pub use crate::vector::XMVector4Orthogonal;
            pub use crate::vector::XMVector4ReciprocalLength;
            pub use crate::vector::XMVector4ReciprocalLengthEst;
            pub use crate::vector::XMVector4Reflect;
            pub use crate::vector::XMVector4Refract;
            pub use crate::vector::XMVector4RefractV;
        }
        pub mod transformation {
            pub use crate::vector::XMVector4Transform;
            // TODO: pub use crate::vector::XMVector4TransformStream;
        }
    }

    /// Vector manipulation functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-templates>
    pub mod template {
        pub use crate::XMMax;
        pub use crate::XMMin;
        pub use crate::XMVectorPermute;
        pub use crate::XMVectorRotateLeft;
        pub use crate::XMVectorRotateRight;
        pub use crate::XMVectorShiftLeft;
        pub use crate::XMVectorSwizzle;
        pub use crate::XMVectorInsert;
    }

    /// Vector accessors
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-accessors>
    pub mod accessor {
        pub use crate::vector::XMVectorGetByIndex;
        // TODO: pub use crate::vector::XMVectorGetByIndexPtr;
        pub use crate::vector::XMVectorGetIntByIndex;
        // TODO: pub use crate::vector::XMVectorGetIntByIndexPtr;
        pub use crate::vector::XMVectorGetIntW;
        // TODO: pub use crate::vector::XMVectorGetIntWPtr;
        pub use crate::vector::XMVectorGetIntX;
        // TODO: pub use crate::vector::XMVectorGetIntXPtr;
        pub use crate::vector::XMVectorGetIntY;
        // TODO: pub use crate::vector::XMVectorGetIntYPtr;
        pub use crate::vector::XMVectorGetIntZ;
        // TODO: pub use crate::vector::XMVectorGetIntZPtr;
        pub use crate::vector::XMVectorGetW;
        pub use crate::vector::XMVectorGetWPtr;
        pub use crate::vector::XMVectorGetX;
        pub use crate::vector::XMVectorGetXPtr;
        pub use crate::vector::XMVectorGetY;
        pub use crate::vector::XMVectorGetYPtr;
        pub use crate::vector::XMVectorGetZ;
        pub use crate::vector::XMVectorGetZPtr;
        pub use crate::vector::XMVectorSetByIndex;
        // TODO: pub use crate::vector::XMVectorSetByIndexPtr;
        pub use crate::vector::XMVectorSetIntByIndex;
        // TODO: pub use crate::vector::XMVectorSetIntByIndexPtr;
        pub use crate::vector::XMVectorSetIntW;
        // TODO: pub use crate::vector::XMVectorSetIntWPtr;
        pub use crate::vector::XMVectorSetIntX;
        // TODO: pub use crate::vector::XMVectorSetIntXPtr;
        pub use crate::vector::XMVectorSetIntY;
        // TODO: pub use crate::vector::XMVectorSetIntYPtr;
        pub use crate::vector::XMVectorSetIntZ;
        // TODO: pub use crate::vector::XMVectorSetIntZPtr;
        pub use crate::vector::XMVectorSetW;
        // TODO: pub use crate::vector::XMVectorSetWPtr;
        pub use crate::vector::XMVectorSetX;
        // TODO: pub use crate::vector::XMVectorSetXPtr;
        pub use crate::vector::XMVectorSetY;
        // TODO: pub use crate::vector::XMVectorSetYPtr;
        pub use crate::vector::XMVectorSetZ;
        // TODO: pub use crate::vector::XMVectorSetZPtr;
    }

    /// Vector and Matrix load functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-load>
    pub mod load {
        // TODO: pub use crate::convert::XMLoadByte2;
        // TODO: pub use crate::convert::XMLoadByte4;
        // TODO: pub use crate::convert::XMLoadByteN2;
        // TODO: pub use crate::convert::XMLoadByteN4;
        // TODO: pub use crate::convert::XMLoadColor;
        // TODO: pub use crate::convert::XMLoadDec4;
        // TODO: pub use crate::convert::XMLoadDecN4;
        // TODO: pub use crate::convert::XMLoadFloat;
        pub use crate::convert::XMLoadFloat2;
        // TODO: pub use crate::convert::XMLoadFloat2A;
        pub use crate::convert::XMLoadFloat3;
        // TODO: pub use crate::convert::XMLoadFloat3A;
        // TODO: pub use crate::convert::XMLoadFloat3PK;
        // TODO: pub use crate::convert::XMLoadFloat3SE;
        pub use crate::convert::XMLoadFloat3x3;
        // TODO: pub use crate::convert::XMLoadFloat3x4;
        // TODO: pub use crate::convert::XMLoadFloat3x4A;
        pub use crate::convert::XMLoadFloat4;
        // TODO: pub use crate::convert::XMLoadFloat4A;
        // TODO: pub use crate::convert::XMLoadFloat4x3;
        // TODO: pub use crate::convert::XMLoadFloat4x3A;
        pub use crate::convert::XMLoadFloat4x4;
        // TODO: pub use crate::convert::XMLoadFloat4x4A;
        // TODO: pub use crate::convert::XMLoadHalf2;
        // TODO: pub use crate::convert::XMLoadHalf4;
        // TODO: pub use crate::convert::XMLoadInt;
        pub use crate::convert::XMLoadInt2;
        // TODO: pub use crate::convert::XMLoadInt2A;
        pub use crate::convert::XMLoadInt3;
        // TODO: pub use crate::convert::XMLoadInt3A;
        pub use crate::convert::XMLoadInt4;
        // TODO: pub use crate::convert::XMLoadInt4A;
        // TODO: pub use crate::convert::XMLoadShort2;
        // TODO: pub use crate::convert::XMLoadShort4;
        // TODO: pub use crate::convert::XMLoadShortN2;
        // TODO: pub use crate::convert::XMLoadShortN4;
        // TODO: pub use crate::convert::XMLoadSInt2;
        // TODO: pub use crate::convert::XMLoadSInt3;
        // TODO: pub use crate::convert::XMLoadSInt4;
        // TODO: pub use crate::convert::XMLoadU555;
        // TODO: pub use crate::convert::XMLoadU565;
        // TODO: pub use crate::convert::XMLoadUByte2;
        // TODO: pub use crate::convert::XMLoadUByte4;
        // TODO: pub use crate::convert::XMLoadUByteN2;
        // TODO: pub use crate::convert::XMLoadUByteN4;
        // TODO: pub use crate::convert::XMLoadUDec4;
        // TODO: pub use crate::convert::XMLoadUDecN4;
        // TODO: pub use crate::convert::XMLoadUDecN4_XR;
        // TODO: pub use crate::convert::XMLoadUInt2;
        // TODO: pub use crate::convert::XMLoadUInt3;
        // TODO: pub use crate::convert::XMLoadUInt4;
        // TODO: pub use crate::convert::XMLoadUNibble4;
        // TODO: pub use crate::convert::XMLoadUShort2;
        // TODO: pub use crate::convert::XMLoadUShort4;
        // TODO: pub use crate::convert::XMLoadUShortN2;
        // TODO: pub use crate::convert::XMLoadUShortN4;
        // TODO: pub use crate::convert::XMLoadXDec4;
        // TODO: pub use crate::convert::XMLoadXDecN4;
    }

    /// Vector and Matrix store functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-functions-storage>
    pub mod store {
        // TODO: pub use crate::convert::XMStoreByte2;
        // TODO: pub use crate::convert::XMStoreByte4;
        // TODO: pub use crate::convert::XMStoreByteN2;
        // TODO: pub use crate::convert::XMStoreByteN4;
        // TODO: pub use crate::convert::XMStoreColor;
        // TODO: pub use crate::convert::XMStoreDec4;
        // TODO: pub use crate::convert::XMStoreDecN4;
        // TODO: pub use crate::convert::XMStoreFloat;
        pub use crate::convert::XMStoreFloat2;
        // TODO: pub use crate::convert::XMStoreFloat2A;
        pub use crate::convert::XMStoreFloat3;
        // TODO: pub use crate::convert::XMStoreFloat3A;
        // TODO: pub use crate::convert::XMStoreFloat3PK;
        // TODO: pub use crate::convert::XMStoreFloat3SE;
        pub use crate::convert::XMStoreFloat3x3;
        // TODO: pub use crate::convert::XMStoreFloat3x4;
        // TODO: pub use crate::convert::XMStoreFloat3x4A;
        pub use crate::convert::XMStoreFloat4;
        // TODO: pub use crate::convert::XMStoreFloat4A;
        // TODO: pub use crate::convert::XMStoreFloat4x3;
        // TODO: pub use crate::convert::XMStoreFloat4x3A;
        pub use crate::convert::XMStoreFloat4x4;
        // TODO: pub use crate::convert::XMStoreFloat4x4A;
        // TODO: pub use crate::convert::XMStoreHalf2;
        // TODO: pub use crate::convert::XMStoreHalf4;
        // TODO: pub use crate::convert::XMStoreInt;
        pub use crate::convert::XMStoreInt2;
        // TODO: pub use crate::convert::XMStoreInt2A;
        pub use crate::convert::XMStoreInt3;
        // TODO: pub use crate::convert::XMStoreInt3A;
        pub use crate::convert::XMStoreInt4;
        // TODO: pub use crate::convert::XMStoreInt4A;
        // TODO: pub use crate::convert::XMStoreShort2;
        // TODO: pub use crate::convert::XMStoreShort4;
        // TODO: pub use crate::convert::XMStoreShortN2;
        // TODO: pub use crate::convert::XMStoreShortN4;
        // TODO: pub use crate::convert::XMStoreSInt2;
        // TODO: pub use crate::convert::XMStoreSInt3;
        // TODO: pub use crate::convert::XMStoreSInt4;
        // TODO: pub use crate::convert::XMStoreU555;
        // TODO: pub use crate::convert::XMStoreU565;
        // TODO: pub use crate::convert::XMStoreUByte2;
        // TODO: pub use crate::convert::XMStoreUByte4;
        // TODO: pub use crate::convert::XMStoreUByteN2;
        // TODO: pub use crate::convert::XMStoreUByteN4;
        // TODO: pub use crate::convert::XMStoreUDec4;
        // TODO: pub use crate::convert::XMStoreUDecN4;
        // TODO: pub use crate::convert::XMStoreUDecN4_XR;
        // TODO: pub use crate::convert::XMStoreUInt2;
        // TODO: pub use crate::convert::XMStoreUInt3;
        // TODO: pub use crate::convert::XMStoreUInt4;
        // TODO: pub use crate::convert::XMStoreUNibble4;
        // TODO: pub use crate::convert::XMStoreUShort2;
        // TODO: pub use crate::convert::XMStoreUShort4;
        // TODO: pub use crate::convert::XMStoreUShortN2;
        // TODO: pub use crate::convert::XMStoreUShortN4;
        // TODO: pub use crate::convert::XMStoreXDec4;
        // TODO: pub use crate::convert::XMStoreXDecN4;
    }

    /// Utility and comparison functions
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-utilities>
    pub mod utility {
        pub use crate::XMComparisonAllFalse;
        pub use crate::XMComparisonAllInBounds;
        pub use crate::XMComparisonAllTrue;
        pub use crate::XMComparisonAnyFalse;
        pub use crate::XMComparisonAnyOutOfBounds;
        pub use crate::XMComparisonAnyTrue;
        pub use crate::XMComparisonMixed;
        pub use crate::misc::XMFresnelTerm;
        // TODO: pub use crate::XMVerifyCPUSupport;
    }

    // pub mod structures {
    //     // TODO
    // }

    // /// Collision classes
    // ///
    // /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-directxmath-classes>
    // pub mod classes {
    //     // TODO: pub use crate::collision::BoundingBox;
    //     // TODO: pub use crate::collision::BoundingFrustum;
    //     // TODO: pub use crate::collision::BoundingOrientedBox;
    //     // TODO: pub use crate::collision::BoundingSphere;
    // }

    // pub mod enumerations {
    //     // TODO
    // }
    // pub mod types {
    //     // TODO
    // }
}

pub use doc::*;

pub const XM_PI: f32 = 3.141592654;
pub const XM_2PI: f32 = 6.283185307;
pub const XM_1DIVPI: f32 = 0.318309886;
pub const XM_1DIV2PI: f32 = 0.159154943;
pub const XM_PIDIV2: f32 = 1.570796327;
pub const XM_PIDIV4: f32 = 0.785398163;

pub const XM_SELECT_0: u32 = 0x00000000;
pub const XM_SELECT_1: u32 = 0xFFFFFFFF;

pub const XM_PERMUTE_0X: u32 = 0;
pub const XM_PERMUTE_0Y: u32 = 1;
pub const XM_PERMUTE_0Z: u32 = 2;
pub const XM_PERMUTE_0W: u32 = 3;
pub const XM_PERMUTE_1X: u32 = 4;
pub const XM_PERMUTE_1Y: u32 = 5;
pub const XM_PERMUTE_1Z: u32 = 6;
pub const XM_PERMUTE_1W: u32 = 7;

pub const XM_SWIZZLE_X: u32 = 0;
pub const XM_SWIZZLE_Y: u32 = 1;
pub const XM_SWIZZLE_Z: u32 = 2;
pub const XM_SWIZZLE_W: u32 = 3;

pub const XM_CRMASK_CR6: u32 = 0x0000000;
pub const XM_CRMASK_CR6TRUE: u32 = 0x00000080;
pub const XM_CRMASK_CR6FALSE: u32 = 0x00000020;
pub const XM_CRMASK_CR6BOUNDS: u32 = XM_CRMASK_CR6FALSE;

pub const XM_CACHE_LINE_SIZE: u32 = 64;

// Type defs to allow the XMVectorSwizzle and XMVectorPermute translation
// to look more like the source

pub type XM_SWIZZLE_X = SwizzleX;
pub type XM_SWIZZLE_Y = SwizzleY;
pub type XM_SWIZZLE_Z = SwizzleZ;
pub type XM_SWIZZLE_W = SwizzleW;

pub type XM_PERMUTE_0X = Permute0X;
pub type XM_PERMUTE_0Y = Permute0Y;
pub type XM_PERMUTE_0Z = Permute0Z;
pub type XM_PERMUTE_0W = Permute0W;
pub type XM_PERMUTE_1X = Permute1X;
pub type XM_PERMUTE_1Y = Permute1Y;
pub type XM_PERMUTE_1Z = Permute1Z;
pub type XM_PERMUTE_1W = Permute1W;

/// Converts an angle measured in degrees into one measured in radians.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertToRadians>
#[inline]
pub fn XMConvertToRadians(fDegrees: f32) -> f32 { return fDegrees * (XM_PI / 180.0); }

/// Converts an angle measured in radians into one measured in degrees.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertToDegrees>
#[inline]
pub fn XMConvertToDegrees(fRadians: f32) -> f32 { return fRadians * (180.0 / XM_PI); }

/// Tests the comparison value to determine if all of the compared components are true.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonAllTrue>
#[inline]
pub fn XMComparisonAllTrue(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6TRUE) == XM_CRMASK_CR6TRUE); }

/// Tests the comparison value to determine if any of the compared components are true.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonAnyTrue>
#[inline]
pub fn XMComparisonAnyTrue(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6FALSE) != XM_CRMASK_CR6FALSE); }

/// Tests the comparison value to determine if all of the compared components are false.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonAllFalse>
#[inline]
pub fn XMComparisonAllFalse(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6FALSE) == XM_CRMASK_CR6FALSE); }

/// Tests the comparison value to determine if any of the compared components are false.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonAnyFalse>
#[inline]
pub fn XMComparisonAnyFalse(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6TRUE) != XM_CRMASK_CR6TRUE); }

/// Tests the comparison value to determine if the compared components had mixed results--some true and some false.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonMixed>
#[inline]
pub fn XMComparisonMixed(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6) == 0); }

/// Tests the comparison value to determine if all of the compared components are within set bounds.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonAllInBounds>
#[inline]
pub fn XMComparisonAllInBounds(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6BOUNDS) == XM_CRMASK_CR6BOUNDS); }

/// Tests the comparison value to determine if any of the compared components are outside the set bounds.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMComparisonAnyOutOfBounds>
#[inline]
pub fn XMComparisonAnyOutOfBounds(CR: u32) -> bool { return (((CR)&XM_CRMASK_CR6BOUNDS) != XM_CRMASK_CR6BOUNDS); }

#[cfg(_XM_NO_INTRINSICS_)]
#[derive(Copy, Clone)]
#[repr(C, align(16))]
#[doc(hidden)]
pub union __vector4 {
    vector4_f32: [f32; 4],
    vector4_u32: [u32; 4],
}

#[cfg(_XM_NO_INTRINSICS_)]
impl std::fmt::Debug for __vector4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unsafe {
            f.debug_struct("__vector4")
                .field("x", &self.vector4_f32[0])
                .field("y", &self.vector4_f32[1])
                .field("z", &self.vector4_f32[2])
                .field("w", &self.vector4_f32[3])
                .finish()
        }
    }
}

#[cfg(_XM_SSE_INTRINSICS_)]
pub type XMVECTOR = __m128;

#[cfg(_XM_ARM_NEON_INTRINSICS_)]
pub type XMVECTOR = float32x4_t;

#[cfg(_XM_NO_INTRINSICS_)]
pub type XMVECTOR = __vector4;

pub type FXMVECTOR = XMVECTOR;
pub type GXMVECTOR = XMVECTOR;
pub type HXMVECTOR = XMVECTOR;
pub type CXMVECTOR<'a> = &'a XMVECTOR;

macro_rules! cast_m128 {
    ($Name:ident) => {
        impl $Name {
            #[cfg(_XM_SSE_INTRINSICS_)]
            #[allow(dead_code)]
            #[inline]
            fn m128i(self) -> __m128i {
                unsafe {
                    _mm_castps_si128(self.v)
                }
            }

            #[cfg(_XM_SSE_INTRINSICS_)]
            #[allow(dead_code)]
            #[inline]
            fn m128d(self) -> __m128d {
                unsafe {
                    _mm_castps_pd(self.v)
                }
            }

            #[cfg(_XM_SSE_INTRINSICS_)]
            #[allow(dead_code)]
            #[inline]
            fn m128(self) -> __m128 {
                unsafe {
                    self.v
                }
            }
        }

        impl std::ops::Deref for $Name {
            type Target = XMVECTOR;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { mem::transmute(self) }
            }
        }

        // Adding Deref is intended to make access to constants
        // easier. Is there a valid use case for DerefMut?
        // impl std::ops::DerefMut for $Name {
        //     fn deref_mut(&mut self) -> &mut Self::Target {
        //         unsafe { mem::transmute(self) }
        //     }
        // }
    }
}

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMVECTORF32 {
    pub f: [f32; 4],
    pub v: XMVECTOR,
}
cast_m128!(XMVECTORF32);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMVECTORI32 {
    pub i: [i32; 4],
    pub v: XMVECTOR,
}
cast_m128!(XMVECTORI32);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMVECTORU8 {
    pub u: [u8; 16],
    pub v: XMVECTOR,
}
cast_m128!(XMVECTORU8);

#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMVECTORU32 {
    pub u: [u32; 4],
    pub v: XMVECTOR,
}
cast_m128!(XMVECTORU32);

/// Unit struct for [`XMVECTOR`] operator overloads.
///
/// [`XMVECTOR`]: type.XMVECTOR.html
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct XMVector(pub XMVECTOR);

// #[cfg(_XM_NO_INTRINSICS_)]
// #[derive(Copy, Clone, Debug)]
// #[doc(hidden)]
// struct RowColumn {
//     pub _11: f32,
//     pub _12: f32,
//     pub _13: f32,
//     pub _14: f32,
//     pub _21: f32,
//     pub _22: f32,
//     pub _23: f32,
//     pub _24: f32,
//     pub _31: f32,
//     pub _32: f32,
//     pub _33: f32,
//     pub _34: f32,
//     pub _41: f32,
//     pub _42: f32,
//     pub _43: f32,
//     pub _44: f32,
// }

// #[cfg(_XM_NO_INTRINSICS_)]
// #[derive(Copy, Clone)]
// #[repr(C, align(16))]
// pub union XMMATRIX {
//     pub r: [XMVECTOR; 4],
//     #[doc(hidden)]
//     pub m: [[f32; 4]; 4],
//     #[doc(hidden)]
//     pub mm: mm,
// }

//#[cfg(not(_XM_NO_INTRINSICS_))]
#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMMATRIX {
    pub r: [XMVECTOR; 4],
    #[cfg(_XM_NO_INTRINSICS_)]
    m: [[f32; 4]; 4],
    // #[cfg(_XM_NO_INTRINSICS_)]
    // rc: RowColumn,
}

impl std::fmt::Debug for XMMATRIX {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        XMMatrix(*self).fmt(f)
    }
}

pub type FXMMATRIX = XMMATRIX;
pub type CXMMATRIX<'a> = &'a XMMATRIX;

/// Unit struct for [`XMMATRIX`] operator overloads.
///
/// [`XMMATRIX`]: union.XMMATRIX.html
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct XMMatrix(pub XMMATRIX);


macro_rules! xm_struct {
    ($Name:ident, $type:ty, $length:expr, $($field:ident),*) => {
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(C)]
        pub struct $Name {
            $(
                pub $field: $type,
            )*
        }

        impl From<[$type; $length]> for $Name {
            fn from(a: [$type; $length]) -> $Name {
                unsafe { std::mem::transmute(a) }
            }
        }

        impl Into<[$type; $length]> for $Name {
            fn into(self) -> [$type; $length] {
                unsafe { std::mem::transmute(self) }
            }
        }

        impl<'a> From<&'a [$type; $length]> for &'a $Name {
            fn from(a: &'a [$type; $length]) -> &'a $Name {
                unsafe { std::mem::transmute(a) }
            }
        }

        impl<'a> Into<&'a [$type; $length]> for &'a $Name {
            fn into(self) -> &'a [$type; $length] {
                unsafe { std::mem::transmute(self) }
            }
        }

        // impl<'a> $Name {
        //     pub fn from_raw_slice(a: &'a [[$type; $length]]) -> &'a [$Name] {
        //         unsafe { std::mem::transmute(a) }
        //     }

        //     pub fn from_raw_slice_mut(a: &'a mut [[$type; $length]]) -> &'a mut [$Name] {
        //         unsafe { std::mem::transmute(a) }
        //     }

        //     pub fn into_raw_slice(a: &'a [$Name]) -> &'a [[$type; $length]] {
        //         unsafe { std::mem::transmute(a) }
        //     }

        //     pub fn into_raw_slice_mut(a: &'a mut [$Name]) -> &'a mut [[$type; $length]] {
        //         unsafe { std::mem::transmute(a) }
        //     }
        // }
    };
}


xm_struct!(XMINT2, i32, 2, x, y);
xm_struct!(XMINT3, i32, 3, x, y, z);
xm_struct!(XMINT4, i32, 4, x, y, z, w);

xm_struct!(XMUINT2, u32, 2, x, y);
xm_struct!(XMUINT3, u32, 3, x, y, z);
xm_struct!(XMUINT4, u32, 4, x, y, z, w);

xm_struct!(XMFLOAT2, f32, 2, x, y);
xm_struct!(XMFLOAT3, f32, 3, x, y, z);
xm_struct!(XMFLOAT4, f32, 4, x, y, z, w);

// #[derive(Copy, Clone, Debug, Default)]
// #[repr(C, align(16))]
// pub struct XMFLOAT2A {
//     pub x: f32,
//     pub y: f32,
// }

// impl From<[f32; 2]> for XMFLOAT2A {
//     fn from(a: [f32; 2]) -> XMFLOAT2A {
//         XMFLOAT2A {
//             x: a[0],
//             y: a[1],
//         }
//     }
// }

// impl Into<[f32; 2]> for XMFLOAT2A {
//     fn into(self) -> [f32; 2] {
//         [self.x, self.y]
//     }
// }

// #[derive(Copy, Clone, Debug, Default)]
// #[repr(C, align(16))]
// pub struct XMFLOAT3A {
//     pub x: f32,
//     pub y: f32,
//     pub z: f32,
// }

// impl From<[f32; 3]> for XMFLOAT3A {
//     fn from(a: [f32; 3]) -> XMFLOAT3A {
//         XMFLOAT3A {
//             x: a[0],
//             y: a[1],
//             z: a[2],
//         }
//     }
// }

// impl Into<[f32; 3]> for XMFLOAT3A {
//     fn into(self) -> [f32; 3] {
//         [self.x, self.y, self.z]
//     }
// }

// #[derive(Copy, Clone, Debug, Default)]
// #[repr(C, align(16))]
// pub struct XMFLOAT4A {
//     pub x: f32,
//     pub y: f32,
//     pub z: f32,
//     pub w: f32,
// }

// impl From<[f32; 4]> for XMFLOAT4A {
//     fn from(a: [f32; 4]) -> XMFLOAT4A {
//         XMFLOAT4A {
//             x: a[0],
//             y: a[1],
//             z: a[2],
//             w: a[3],
//         }
//     }
// }

// impl Into<[f32; 4]> for XMFLOAT4A {
//     fn into(self) -> [f32; 4] {
//         [self.x, self.y, self.z, self.w]
//     }
// }

// TODO: XMFLOAT4X3
// TODO: XMFLOAT4X3A
// TODO: XMFLOAT3X4
// TODO: XMFLOAT3X4A

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct XMFLOAT4X4 {
    pub m: [[f32; 4]; 4],
}

// [f32; 16]

impl From<[f32; 16]> for XMFLOAT4X4 {
    fn from(a: [f32; 16]) -> XMFLOAT4X4 {
        unsafe { std::mem::transmute(a) }
    }
}

impl Into<[f32; 16]> for XMFLOAT4X4 {
    fn into(self) -> [f32; 16] {
        unsafe { std::mem::transmute(self) }
    }
}

impl<'a> From<&'a [f32; 16]> for &'a XMFLOAT4X4 {
    fn from(a: &'a [f32; 16]) -> &'a XMFLOAT4X4 {
        unsafe { std::mem::transmute(a) }
    }
}

impl<'a> Into<&'a [f32; 16]> for &'a XMFLOAT4X4 {
    fn into(self) -> &'a [f32; 16] {
        unsafe { std::mem::transmute(self) }
    }
}

// [[f32; 4]; 4]

impl From<[[f32; 4]; 4]> for XMFLOAT4X4 {
    fn from(a: [[f32; 4]; 4]) -> XMFLOAT4X4 {
        unsafe { std::mem::transmute(a) }
    }
}

impl Into<[[f32; 4]; 4]> for XMFLOAT4X4 {
    fn into(self) -> [[f32; 4]; 4] {
        unsafe { std::mem::transmute(self) }
    }
}

impl<'a> From<&'a [[f32; 4]; 4]> for &'a XMFLOAT4X4 {
    fn from(a: &'a [[f32; 4]; 4]) -> &'a XMFLOAT4X4 {
        unsafe { std::mem::transmute(a) }
    }
}

impl<'a> Into<&'a [[f32; 4]; 4]> for &'a XMFLOAT4X4 {
    fn into(self) -> &'a [[f32; 4]; 4] {
        unsafe { std::mem::transmute(self) }
    }
}

// #[derive(Copy, Clone, Debug, Default)]
// #[repr(C, align(16))]
// pub struct XMFLOAT4X4A {
//     pub m: [[f32; 4]; 4],
// }

// TODO: XMFLOAT4X4A From/Into

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct XMFLOAT3X3 {
    pub m: [[f32; 3]; 3],
}

// TODO: XMFLOAT3X3 From/Into

// #[derive(Copy, Clone, Debug, Default)]
// #[repr(C, align(16))]
// pub struct XMFLOAT3X3A {
//     pub m: [[f32; 3]; 3],
// }

// TODO: XMFLOAT3X3A From/Into


/// Compares two numeric data type instances, or two instances of an object
/// which supports an overload of `<`, and returns the smaller one of the two
/// instances. The data type of the arguments and the return value is the same.
///
/// <https://docs.microsoft.com/en-us/windows/win32/dxmath/xmmin-template>
#[inline]
pub fn XMMin<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

/// Compares two numeric data type instances, or two instances of an object
/// which supports an overload of `<`, and returns the larger one of the two
/// instances. The data type of the arguments and the return value is the same.
///
/// <https://docs.microsoft.com/en-us/windows/win32/dxmath/xmmax-template>
#[inline]
pub fn XMMax<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

// const generics:
// https://github.com/rust-lang/rust/issues/44580
//
// specializaiton:
// https://github.com/rust-lang/rust/issues/31844
//
// specialization work-around on stable:
// https://github.com/dtolnay/case-studies/blob/master/autoref-specialization/README.md
//
// Note that the specialization work-wround doesn't quite work when there is no
// self receiver.

// TODO: Internal / PermuteHelper template

/// XMVectorPermute trait parameters
pub unsafe trait Permute {
    const PERMUTE: u32;
}

pub enum Permute0X {}
pub enum Permute0Y {}
pub enum Permute0Z {}
pub enum Permute0W {}

pub enum Permute1X {}
pub enum Permute1Y {}
pub enum Permute1Z {}
pub enum Permute1W {}

unsafe impl Permute for Permute0X {
    const PERMUTE: u32 = XM_PERMUTE_0X;
}

unsafe impl Permute for Permute0Y {
    const PERMUTE: u32 = XM_PERMUTE_0Y;
}

unsafe impl Permute for Permute0Z {
    const PERMUTE: u32 = XM_PERMUTE_0Z;
}

unsafe impl Permute for Permute0W {
    const PERMUTE: u32 = XM_PERMUTE_0W;
}

unsafe impl Permute for Permute1X {
    const PERMUTE: u32 = XM_PERMUTE_1X;
}

unsafe impl Permute for Permute1Y {
    const PERMUTE: u32 = XM_PERMUTE_1Y;
}

unsafe impl Permute for Permute1Z {
    const PERMUTE: u32 = XM_PERMUTE_1Z;
}

unsafe impl Permute for Permute1W {
    const PERMUTE: u32 = XM_PERMUTE_1W;
}

/// Permutes the components of two vectors to create a new vector.
pub trait XMVectorPermute {
    #[doc(hidden)]
    const WhichX: bool;
    #[doc(hidden)]
    const WhichY: bool;
    #[doc(hidden)]
    const WhichZ: bool;
    #[doc(hidden)]
    const WhichW: bool;
    #[doc(hidden)]
    const Shuffle: i32;
    #[doc(hidden)]
    const SelectMask: XMVECTORU32 = {
        // https://github.com/rust-lang/rust/issues/49146
        // XMVECTORU32 {
        //     u: [
        //         if Self::WhichX { 0xFFFFFFFF } else { 0 },
        //         if Self::WhichY { 0xFFFFFFFF } else { 0 },
        //         if Self::WhichZ { 0xFFFFFFFF } else { 0 },
        //         if Self::WhichW { 0xFFFFFFFF } else { 0 },
        //     ]
        // };

        // https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
        const fn mask(which: bool) -> u32 {
            let f = which as u32;
            let m = 0xFFFFFFFF;
            (-(f as i32) as u32 & m) | !m
        }

        XMVECTORU32 {
            u: [
                mask(Self::WhichX),
                mask(Self::WhichY),
                mask(Self::WhichZ),
                mask(Self::WhichW),
            ]
        }
    };

    /// Permutes the components of two vectors to create a new vector.
    ///
    /// # Remarks
    ///
    /// This function is a template version of [`XMVectorPermute`] where the `Permute*` arguments are template values.
    ///
    /// ```rust
    /// # use directx_math::*;
    /// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
    /// let b = XMVectorSet(5.0, 6.0, 7.0, 8.0);
    ///
    /// let c = <(Permute0X, Permute0Z, Permute1X, Permute1Z)>::XMVectorPermute(a, b);
    /// let d = XMVectorSet(1.0, 3.0, 5.0, 7.0);
    ///
    /// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
    /// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
    /// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
    /// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
    /// ```
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/xmvectorpermute-template>
    ///
    /// [`XMVectorPermute`]: vector/fn.XMVectorPermute.html
    fn XMVectorPermute(V1: XMVECTOR, V2: XMVECTOR) -> XMVECTOR;
}

#[cfg(not(nightly_specialization))]
impl<X: Permute, Y: Permute, Z: Permute, W: Permute> XMVectorPermute for (X, Y, Z, W) {
    const WhichX: bool = X::PERMUTE > 3;
    const WhichY: bool = Y::PERMUTE > 3;
    const WhichZ: bool = Z::PERMUTE > 3;
    const WhichW: bool = W::PERMUTE > 3;
    const Shuffle: i32 = _MM_SHUFFLE(W::PERMUTE & 3, Z::PERMUTE & 3, Y::PERMUTE & 3, X::PERMUTE & 3);

    #[inline]
    fn XMVectorPermute(V1: XMVECTOR, V2: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            let shuffled1: XMVECTOR = XM_PERMUTE_PS!(V1, Self::Shuffle);
            let shuffled2: XMVECTOR = XM_PERMUTE_PS!(V2, Self::Shuffle);

            let masked1: XMVECTOR = _mm_andnot_ps(Self::SelectMask.v, shuffled1);
            let masked2: XMVECTOR = _mm_and_ps(Self::SelectMask.v, shuffled2);

            return _mm_or_ps(masked1, masked2);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            XMVectorPermute(V1, V2, X::PERMUTE, Y::PERMUTE, Z::PERMUTE, W::PERMUTE)
        }
    }
}

// This is only implemented as a macro to hide it from rust-analyzer until
// the specialization bug is fixed.
//
// https://github.com/rust-analyzer/rust-analyzer/issues/4264
macro_rules! XMVectorPermute_nightly_specialization {
    () => {
        #[cfg(nightly_specialization)]
        impl<X: Permute, Y: Permute, Z: Permute, W: Permute> XMVectorPermute for (X, Y, Z, W) {
            default const WhichX: bool = X::PERMUTE > 3;
            default const WhichY: bool = Y::PERMUTE > 3;
            default const WhichZ: bool = Z::PERMUTE > 3;
            default const WhichW: bool = W::PERMUTE > 3;
            default const Shuffle: i32 = _MM_SHUFFLE(W::PERMUTE & 3, Z::PERMUTE & 3, Y::PERMUTE & 3, X::PERMUTE & 3);

            #[inline]
            default fn XMVectorPermute(V1: XMVECTOR, V2: XMVECTOR) -> XMVECTOR {
                #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
                unsafe {
                    let shuffled1: XMVECTOR = XM_PERMUTE_PS!(V1, Self::Shuffle);
                    let shuffled2: XMVECTOR = XM_PERMUTE_PS!(V2, Self::Shuffle);

                    let masked1: XMVECTOR = _mm_andnot_ps(Self::SelectMask.v, shuffled1);
                    let masked2: XMVECTOR = _mm_and_ps(Self::SelectMask.v, shuffled2);

                    return _mm_or_ps(masked1, masked2);
                }

                #[cfg(_XM_ARM_NEON_INTRINSICS_)]
                {
                    unimplemented!()
                }

                #[cfg(_XM_NO_INTRINSICS_)]
                {
                    XMVectorPermute(V1, V2, X::PERMUTE, Y::PERMUTE, Z::PERMUTE, W::PERMUTE)
                }
            }
        }
    }
}

XMVectorPermute_nightly_specialization!();

#[test]
fn test_XMVectorPermuteTrait() {
    let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
    let b = XMVectorSet(5.0, 6.0, 7.0, 8.0);

    let c = <(Permute0X, Permute0Z, Permute1X, Permute1Z)>::XMVectorPermute(a, b);
    let d = XMVectorSet(1.0, 3.0, 5.0, 7.0);

    assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
    assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
    assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
    assert_eq!(XMVectorGetW(c), XMVectorGetW(d));

    let e = <(Permute0Y, Permute0W, Permute1Y, Permute1W)>::XMVectorPermute(a, b);
    let f = XMVectorSet(2.0, 4.0, 6.0, 8.0);

    assert_eq!(XMVectorGetX(e), XMVectorGetX(f));
    assert_eq!(XMVectorGetY(e), XMVectorGetY(f));
    assert_eq!(XMVectorGetZ(e), XMVectorGetZ(f));
    assert_eq!(XMVectorGetW(e), XMVectorGetW(f));
}

/// Specialized case
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = XMVectorSet(5.0, 6.0, 7.0, 8.0);
///
/// let c = <(Permute0X, Permute0Y, Permute0Z, Permute0W)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(1.0, 2.0, 3.0, 4.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorPermute for (Permute0X, Permute0Y, Permute0Z, Permute0W) {
    const WhichX: bool = Permute0X::PERMUTE > 3;
    const WhichY: bool = Permute0Y::PERMUTE > 3;
    const WhichZ: bool = Permute0Z::PERMUTE > 3;
    const WhichW: bool = Permute0W::PERMUTE > 3;
    const Shuffle: i32 = _MM_SHUFFLE(Permute0W::PERMUTE & 3, Permute0Z::PERMUTE & 3, Permute0Y::PERMUTE & 3, Permute0X::PERMUTE & 3);

    #[inline(always)]
    fn XMVectorPermute(V1: XMVECTOR, _V2: XMVECTOR) -> XMVECTOR {
        V1
    }
}

/// Specialized case
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = XMVectorSet(5.0, 6.0, 7.0, 8.0);
///
/// let c = <(Permute1X, Permute1Y, Permute1Z, Permute1W)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(5.0, 6.0, 7.0, 8.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorPermute for (Permute1X, Permute1Y, Permute1Z, Permute1W) {
    const WhichX: bool = Permute1X::PERMUTE > 3;
    const WhichY: bool = Permute1Y::PERMUTE > 3;
    const WhichZ: bool = Permute1Z::PERMUTE > 3;
    const WhichW: bool = Permute1W::PERMUTE > 3;
    const Shuffle: i32 = _MM_SHUFFLE(Permute1W::PERMUTE & 3, Permute1Z::PERMUTE & 3, Permute1Y::PERMUTE & 3, Permute1X::PERMUTE & 3);

    #[inline(always)]
    fn XMVectorPermute(_V1: XMVECTOR, V2: XMVECTOR) -> XMVECTOR {
        V2
    }
}

/// Specialized case
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = XMVectorSet(5.0, 6.0, 7.0, 8.0);
///
/// let c = <(Permute0X, Permute0Y, Permute1X, Permute1Y)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(1.0, 2.0, 5.0, 6.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorPermute for (Permute0X, Permute0Y, Permute1X, Permute1Y) {
    const WhichX: bool = Permute0X::PERMUTE > 3;
    const WhichY: bool = Permute0Y::PERMUTE > 3;
    const WhichZ: bool = Permute1X::PERMUTE > 3;
    const WhichW: bool = Permute1Y::PERMUTE > 3;
    const Shuffle: i32 = _MM_SHUFFLE(Permute1Y::PERMUTE & 3, Permute1X::PERMUTE & 3, Permute0Y::PERMUTE & 3, Permute0X::PERMUTE & 3);

    #[inline(always)]
    fn XMVectorPermute(V1: XMVECTOR, V2: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            return _mm_movelh_ps(V1, V2);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            type X = Permute0X;
            type Y = Permute0Y;
            type Z = Permute1X;
            type W = Permute1Y;
            XMVectorPermute(V1, V2, X::PERMUTE, Y::PERMUTE, Z::PERMUTE, W::PERMUTE)
        }
    }
}

/// Specialized case
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = XMVectorSet(5.0, 6.0, 7.0, 8.0);
///
/// let c = <(Permute1Z, Permute1W, Permute0Z, Permute0W)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(7.0, 8.0, 3.0, 4.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorPermute for (Permute1Z, Permute1W, Permute0Z, Permute0W) {
    const WhichX: bool = Permute1Z::PERMUTE > 3;
    const WhichY: bool = Permute1W::PERMUTE > 3;
    const WhichZ: bool = Permute0Z::PERMUTE > 3;
    const WhichW: bool = Permute0W::PERMUTE > 3;
    const Shuffle: i32 = _MM_SHUFFLE(Permute0W::PERMUTE & 3, Permute0Z::PERMUTE & 3, Permute1W::PERMUTE & 3, Permute1Z::PERMUTE & 3);

    #[inline(always)]
    fn XMVectorPermute(V1: XMVECTOR, V2: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            return _mm_movehl_ps(V1, V2);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            type X = Permute1Z;
            type Y = Permute1W;
            type Z = Permute0Z;
            type W = Permute0W;
            XMVectorPermute(V1, V2, X::PERMUTE, Y::PERMUTE, Z::PERMUTE, W::PERMUTE)
        }
    }
}

// NOTE: Most of specialized cases for XMVectorPermute follow an explict pattern
//       that could easily be macroed.
// TODO: XMVectorPermute template specializations: _XM_SSE3_INTRINSICS_
// TODO: XMVectorPermute template specializations: _XM_SSE4_INTRINSICS_
// TODO: XMVectorSwizzle template specializations: _XM_ARM_NEON_INTRINSICS_

/// XMVectorSwizzle trait parameters
pub unsafe trait Swizzle {
    const SWIZZLE: u32;
}

pub enum SwizzleX {}
pub enum SwizzleY {}
pub enum SwizzleZ {}
pub enum SwizzleW {}

unsafe impl Swizzle for SwizzleX {
    const SWIZZLE: u32 = XM_SWIZZLE_X;
}

unsafe impl Swizzle for SwizzleY {
    const SWIZZLE: u32 = XM_SWIZZLE_Y;
}

unsafe impl Swizzle for SwizzleZ {
    const SWIZZLE: u32 = XM_SWIZZLE_Z;
}

unsafe impl Swizzle for SwizzleW {
    const SWIZZLE: u32 = XM_SWIZZLE_W;
}

/// Swizzles a vector
pub trait XMVectorSwizzle {
    /// Swizzles a vector.
    ///
    /// # Remarks
    ///
    /// This function is a template version of [`XMVectorSwizzle`] where the `Swizzle*` arguments are template values.
    ///
    ///
    /// ```rust
    /// # use directx_math::*;
    /// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
    /// let b = <(SwizzleW, SwizzleZ, SwizzleY, SwizzleX)>::XMVectorSwizzle(a);
    /// let c = XMVectorSet(4.0, 3.0, 2.0, 1.0);
    ///
    /// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
    /// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
    /// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
    /// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
    /// ```
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/dxmath/xmvectorswizzle-template>
    ///
    /// [`XMVectorSwizzle`]: vector/fn.XMVectorSwizzle.html
    fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR;
}

impl<X: Swizzle, Y: Swizzle, Z: Swizzle, W: Swizzle> XMVectorSwizzle for (X, Y, Z, W) {
    #[inline(always)]
    #[cfg(not(nightly_specialization))]
    fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            XM_PERMUTE_PS!(V, _MM_SHUFFLE(W::SWIZZLE, Z::SWIZZLE, Y::SWIZZLE, X::SWIZZLE))
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            XMVectorSwizzle(V, X::SWIZZLE, Y::SWIZZLE, Z::SWIZZLE, W::SWIZZLE)
        }
    }

    #[inline(always)]
    #[cfg(nightly_specialization)]
    default fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            XM_PERMUTE_PS!(V, _MM_SHUFFLE(W::SWIZZLE, Z::SWIZZLE, Y::SWIZZLE, X::SWIZZLE))
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            XMVectorSwizzle(V, X::SWIZZLE, Y::SWIZZLE, Z::SWIZZLE, W::SWIZZLE)
        }
    }
}

#[test]
fn test_XMVectorSwizzleTrait() {
    let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
    let b = <(SwizzleW, SwizzleZ, SwizzleY, SwizzleX)>::XMVectorSwizzle(a);
    let c = XMVectorSet(4.0, 3.0, 2.0, 1.0);

    assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
    assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
    assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
    assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
}

/// Specialized case
///
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = <(SwizzleX, SwizzleY, SwizzleX, SwizzleY)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(1.0, 2.0, 1.0, 2.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorSwizzle for (SwizzleX, SwizzleY, SwizzleX, SwizzleY) {
    #[inline(always)]
    fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            return _mm_movelh_ps(V, V);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            type X = SwizzleX;
            type Y = SwizzleY;
            type Z = SwizzleX;
            type W = SwizzleY;
            XMVectorSwizzle(V, X::SWIZZLE, Y::SWIZZLE, Z::SWIZZLE, W::SWIZZLE)
        }
    }
}

/// Specialized case
///
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = <(SwizzleZ, SwizzleW, SwizzleZ, SwizzleW)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(3.0, 4.0, 3.0, 4.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorSwizzle for (SwizzleZ, SwizzleW, SwizzleZ, SwizzleW) {
    #[inline(always)]
    fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            return _mm_movehl_ps(V, V);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            type X = SwizzleZ;
            type Y = SwizzleW;
            type Z = SwizzleZ;
            type W = SwizzleW;
            XMVectorSwizzle(V, X::SWIZZLE, Y::SWIZZLE, Z::SWIZZLE, W::SWIZZLE)
        }
    }
}

/// Specialized case
///
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = <(SwizzleX, SwizzleX, SwizzleY, SwizzleY)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(1.0, 1.0, 2.0, 2.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorSwizzle for (SwizzleX, SwizzleX, SwizzleY, SwizzleY) {
    #[inline(always)]
    fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            return _mm_unpacklo_ps(V, V);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            type X = SwizzleX;
            type Y = SwizzleX;
            type Z = SwizzleY;
            type W = SwizzleY;
            XMVectorSwizzle(V, X::SWIZZLE, Y::SWIZZLE, Z::SWIZZLE, W::SWIZZLE)
        }
    }
}

/// Specialized case
///
/// ```rust
/// # use directx_math::*;
/// let a = XMVectorSet(1.0, 2.0, 3.0, 4.0);
/// let b = <(SwizzleZ, SwizzleZ, SwizzleW, SwizzleW)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(3.0, 3.0, 4.0, 4.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
#[cfg(nightly_specialization)]
impl XMVectorSwizzle for (SwizzleZ, SwizzleZ, SwizzleW, SwizzleW) {
    #[inline(always)]
    fn XMVectorSwizzle(V: XMVECTOR) -> XMVECTOR {
        #[cfg(any(_XM_SSE_INTRINSICS_, _XM_AVX_INTRINSICS_))]
        unsafe {
            return _mm_unpackhi_ps(V, V);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_NO_INTRINSICS_)]
        {
            type X = SwizzleZ;
            type Y = SwizzleZ;
            type Z = SwizzleW;
            type W = SwizzleW;
            XMVectorSwizzle(V, X::SWIZZLE, Y::SWIZZLE, Z::SWIZZLE, W::SWIZZLE)
        }
    }
}

// TODO: XMVectorSwizzle template specializations: _XM_SSE3_INTRINSICS_
// TODO: XMVectorSwizzle template specializations: _XM_AVX2_INTRINSICS_
// TODO: XMVectorSwizzle template specializations: _XM_ARM_NEON_INTRINSICS_

// TODO: XMVectorShiftLeft template
// TODO: XMVectorRotateLeft template
// TODO: XMVectorRotateRight template
// TODO: XMVectorInsert template

/// Creates a vector, each of whose components is either `0.0` or `1.0`.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMVectorSetBinaryConstant>
#[inline]
pub fn XMVectorSetBinaryConstant(C0: u32, C1: u32, C2: u32, C3: u32) -> XMVECTOR {
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut vResult: XMVECTORU32 = mem::MaybeUninit::uninit().assume_init();
        vResult.u[0] = (0 - (C0 & 1)) & 0x3F800000;
        vResult.u[1] = (0 - (C1 & 1)) & 0x3F800000;
        vResult.u[2] = (0 - (C2 & 1)) & 0x3F800000;
        vResult.u[3] = (0 - (C3 & 1)) & 0x3F800000;
        return vResult.v;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        const g_vMask1: XMVECTORU32 = XMVECTORU32 { u: [1, 2, 3, 4] };
        let mut vTemp = _mm_set_epi32(C3 as i32, C2 as i32, C1 as i32, C0 as i32);
        // Mask off the low bits
        vTemp = _mm_and_si128(vTemp, g_vMask1.m128i());
        // 0xFFFFFFFF on true bits
        vTemp = _mm_cmpeq_epi32(vTemp, g_vMask1.m128i());
        // 0xFFFFFFFF -> 1.0f, 0x00000000 -> 0.0f
        vTemp = _mm_and_si128(vTemp, g_XMOne.m128i());
        return _mm_castsi128_ps(vTemp);
    }
}

/// Creates a vector with identical floating-point components. Each component is a constant divided by two raised to an integer exponent.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMVectorSplatConstant>
#[inline]
pub fn XMVectorSplatConstant(IntConstant: i32, DivExponent: u32) -> XMVECTOR {
    debug_assert!(IntConstant >= -16 && IntConstant <= 15);
    debug_assert!(DivExponent < 32);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let V = XMVECTORI32 { i: [IntConstant, IntConstant, IntConstant, IntConstant] };
        return XMConvertVectorIntToFloat(V.v, DivExponent);
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // Splat the int
        let mut vScale: __m128i = _mm_set1_epi32(IntConstant);
        // Convert to a float
        let mut vResult: XMVECTOR = _mm_cvtepi32_ps(vScale);
        // Convert DivExponent into 1.0f/(1<<DivExponent)
        let uScale: u32 = 0x3F800000u32 - (DivExponent << 23);
        // Splat the scalar value (It's really a float)
        vScale = _mm_set1_epi32(uScale as i32);
        // Multiply by the reciprocal (Perform a right shift by DivExponent)
        vResult = _mm_mul_ps(vResult, _mm_castsi128_ps(vScale));
        return vResult;
    }
}

/// Creates a vector with identical integer components.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMVectorSplatConstantInt>
#[inline]
pub fn XMVectorSplatConstantInt(IntConstant: i32) -> XMVECTOR {
    debug_assert!(IntConstant >= -16 && IntConstant <= 15);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let V = XMVECTORI32 {
            i: [IntConstant, IntConstant, IntConstant, IntConstant]
        };
        return V.v;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let V: __m128i = _mm_set1_epi32(IntConstant);
        return _mm_castsi128_ps(V);
    }
}
