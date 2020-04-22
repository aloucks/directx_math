
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

#[allow(unused_imports)]
use std::mem;

#[cfg(all(target_arch="x86_64", not(_XM_NO_INTRINSICS_)))]
#[doc(hidden)]
pub use std::arch::x86_64 as arch;

#[cfg(all(target_arch="x86", not(_XM_NO_INTRINSICS_)))]
#[doc(hidden)]
pub use std::arch::x86 as arch;

#[cfg(all(target_arch="arm", not(_XM_NO_INTRINSICS_)))]
#[doc(hidden)]
pub use std::arch::arm as arch;

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
#[cfg(_XM_NO_INTRINSICS_)]
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
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Align16<T> {
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
    }
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

pub mod vector;
pub mod convert;
pub mod globals;
pub mod misc;
pub mod matrix;

pub use vector::*;
pub use convert::*;
pub use globals::*;
pub use misc::*;
pub use matrix::*;

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

/// Converts an angle measured in degrees into one measured in radians.
///
/// https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertToRadians
#[inline]
pub fn XMConvertToRadians(fDegrees: f32) -> f32 { return fDegrees * (XM_PI / 180.0); }

/// Converts an angle measured in radians into one measured in degrees.
///
/// https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertToDegrees
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

#[cfg(_XM_NO_INTRINSICS_)]
#[derive(Copy, Clone, Debug)]
#[doc(hidden)]
pub struct mm {
    pub _11: f32,
    pub _12: f32,
    pub _13: f32,
    pub _14: f32,
    pub _21: f32,
    pub _22: f32,
    pub _23: f32,
    pub _24: f32,
    pub _31: f32,
    pub _32: f32,
    pub _33: f32,
    pub _34: f32,
    pub _41: f32,
    pub _42: f32,
    pub _43: f32,
    pub _44: f32,
}

#[cfg(_XM_NO_INTRINSICS_)]
#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMMATRIX {
    pub r: [XMVECTOR; 4],
    #[doc(hidden)]
    pub m: [[f32; 4]; 4],
    #[doc(hidden)]
    pub mm: mm,
}

#[cfg(not(_XM_NO_INTRINSICS_))]
#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub union XMMATRIX {
    pub r: [XMVECTOR; 4],
}

pub type FXMMATRIX = XMMATRIX;
pub type CXMMATRIX<'a> = &'a XMMATRIX;

// TODO: XMFLOAT2
// TODO: XMFLOAT2A 
// TODO: XMINT2
// TODO: XMUINT2
// TODO: XMFLOAT3
// TODO: XMFLOAT3A 
// TODO: XMFLOAT3A 
// TODO: XMINT3
// TODO: XMUINT3
// TODO: XMFLOAT4
// TODO: XMFLOAT4A
// TODO: XMINT4
// TODO: XMUINT4
// TODO: XMFLOAT3X3
// TODO: XMFLOAT4X3
// TODO: XMFLOAT4X3A 
// TODO: XMFLOAT3X4
// TODO: XMFLOAT4X4
// TODO: XMFLOAT4X4A

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

// TODO: Internal / PermuteHelper template
// TODO: XMVectorPermute template

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
    /// The trait based version is more efficient and the specialized cases provide additional optimization.
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
            // PERFORMANCE: This should be const, but `if` isn't allowed in const context.
            // let selectMask: XMVECTORU32 = XMVECTORU32 {
            //     u: [
            //         if Self::WhichX { 0xFFFFFFFF } else { 0 },
            //         if Self::WhichY { 0xFFFFFFFF } else { 0 },
            //         if Self::WhichZ { 0xFFFFFFFF } else { 0 },
            //         if Self::WhichW { 0xFFFFFFFF } else { 0 },
            //     ]
            // };

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
/// let c = <&(Permute0X, Permute0Y, Permute0Z, Permute0W)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(1.0, 2.0, 3.0, 4.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
impl XMVectorPermute for &(Permute0X, Permute0Y, Permute0Z, Permute0W) {
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
/// let c = <&(Permute1X, Permute1Y, Permute1Z, Permute1W)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(5.0, 6.0, 7.0, 8.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
impl XMVectorPermute for &(Permute1X, Permute1Y, Permute1Z, Permute1W) {
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
/// let c = <&(Permute0X, Permute0Y, Permute1X, Permute1Y)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(1.0, 2.0, 5.0, 6.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
impl XMVectorPermute for &(Permute0X, Permute0Y, Permute1X, Permute1Y) {
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
/// let c = <&(Permute1Z, Permute1W, Permute0Z, Permute0W)>::XMVectorPermute(a, b);
/// let d = XMVectorSet(7.0, 8.0, 3.0, 4.0);
///
/// assert_eq!(XMVectorGetX(c), XMVectorGetX(d));
/// assert_eq!(XMVectorGetY(c), XMVectorGetY(d));
/// assert_eq!(XMVectorGetZ(c), XMVectorGetZ(d));
/// assert_eq!(XMVectorGetW(c), XMVectorGetW(d));
/// ```
impl XMVectorPermute for &(Permute1Z, Permute1W, Permute0Z, Permute0W) {
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
    /// The trait based version is more efficient and the specialized cases provide additional optimization.
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
/// let b = <&(SwizzleX, SwizzleY, SwizzleX, SwizzleY)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(1.0, 2.0, 1.0, 2.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
impl XMVectorSwizzle for &(SwizzleX, SwizzleY, SwizzleX, SwizzleY) {
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
/// let b = <&(SwizzleZ, SwizzleW, SwizzleZ, SwizzleW)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(3.0, 4.0, 3.0, 4.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
impl XMVectorSwizzle for &(SwizzleZ, SwizzleW, SwizzleZ, SwizzleW) {
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
/// let b = <&(SwizzleX, SwizzleX, SwizzleY, SwizzleY)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(1.0, 1.0, 2.0, 2.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
impl XMVectorSwizzle for &(SwizzleX, SwizzleX, SwizzleY, SwizzleY) {
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
/// let b = <&(SwizzleZ, SwizzleZ, SwizzleW, SwizzleW)>::XMVectorSwizzle(a);
/// let c = XMVectorSet(3.0, 3.0, 4.0, 4.0);
///
/// assert_eq!(XMVectorGetX(b), XMVectorGetX(c));
/// assert_eq!(XMVectorGetY(b), XMVectorGetY(c));
/// assert_eq!(XMVectorGetZ(b), XMVectorGetZ(c));
/// assert_eq!(XMVectorGetW(b), XMVectorGetW(c));
/// ```
impl XMVectorSwizzle for &(SwizzleZ, SwizzleZ, SwizzleW, SwizzleW) {
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

/// Creates a vector, each of whose components is either 0.0f or 1.0f.
///
/// https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMVectorSetBinaryConstant
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
/// https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMVectorSplatConstant
#[inline]
pub fn XMVectorSplatConstant(IntConstant: i32, DivExponent: u32) -> XMVECTOR {
    assert!(IntConstant >= -16 && IntConstant <= 15);
    assert!(DivExponent < 32);

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
/// https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMVectorSplatConstantInt
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
