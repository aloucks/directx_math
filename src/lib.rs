
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(unused_parens)]

#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(dead_code)]

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

#[inline]
pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
pub(crate) fn fabsf(x: f32) -> f32 {
    x.abs()
}

#[inline]
pub(crate) fn floorf(x: f32) -> f32 {
    x.floor()
}

#[inline]
pub(crate) fn ceilf(x: f32) -> f32 {
    x.ceil()
}

#[inline]
pub(crate) fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}

#[inline]
pub(crate) fn sinf(x: f32) -> f32 {
    x.sin()
}

#[inline]
pub(crate) fn cosf(x: f32) -> f32 {
    x.cos()
}

#[inline]
pub(crate) fn tanf(x: f32) -> f32 {
    x.tan()
}

#[inline]
pub(crate) fn asinf(x: f32) -> f32 {
    x.asin()
}

#[inline]
pub(crate) fn acosf(x: f32) -> f32 {
    x.acos()
}

#[inline]
pub(crate) fn atanf(x: f32) -> f32 {
    x.atan()
}

#[inline]
pub(crate) fn atan2f(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

#[inline]
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
const fn ubool(a: u32) -> bool {
    a != 0
}

#[inline(always)]
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

#[cfg(not(_XM_FMA3_INTRINSICS_))]
macro_rules! XM_FMADD_PS {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::arch::_mm_add_ps(_mm_mul_ps(($a), ($b)), ($c))
    }
}

#[cfg(not(_XM_FMA3_INTRINSICS_))]
macro_rules! XM_FNMADD_PS {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::arch::_mm_sub_ps(($c), _mm_mul_ps(($a), ($b)))
    }
}

// --

#[cfg(_XM_AVX_INTRINSICS_)]
macro_rules! XM_PERMUTE_PS {
    ($v:expr, $c:expr) => {
        $crate::arch::_mm_permute_ps(($v), $c)
    }
}

#[cfg(not(_XM_AVX_INTRINSICS_))]
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

pub use vector::*;
pub use convert::*;
pub use globals::*;

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

#[inline]
pub fn XMConvertToRadians(fDegrees: f32) -> f32 { return fDegrees * (XM_PI / 180.0); }

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
pub union __vector4 {
    pub vector4_f32: [f32; 4],
    pub vector4_u32: [u32; 4],
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
            #[inline]
            fn m128i(self) -> __m128i {
                unsafe {
                    _mm_castps_si128(self.v)
                }
            }

            #[cfg(_XM_SSE_INTRINSICS_)]
            #[inline]
            fn m128d(self) -> __m128d {
                unsafe {
                    _mm_castps_pd(self.v)
                }
            }
        }
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

/// Unit struct wrapper for `Add`, `AddAssign`, `Mul`, `MulAssign`, `Div`, 
/// and `DivAssign` [XMVECTOR] overloads.
/// 
/// [XMVECTOR]: type.XMVECTOR.html
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct XMVector(pub XMVECTOR);

#[cfg(_XM_NO_INTRINSICS_)]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
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
#[repr(C)]
pub union XMMATRIX {
    pub r: [XMVECTOR; 4],
    pub m: [[f32; 4]; 4],
    pub mm: mm,
}

#[cfg(not(_XM_NO_INTRINSICS_))]
#[derive(Copy, Clone)]
#[repr(C)]
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

#[inline]
pub fn XMMin<T: PartialOrd>(a: T, b: T) -> T { 
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn XMMax<T: PartialOrd>(a: T, b: T) -> T { 
    if a > b {
        a
    } else {
        b
    }
}

// TODO: Internal / PermuteHelper template
// TODO: XMVectorPermute template
// TODO: XMVectorSwizzle template
// TODO: XMVectorShiftLeft template
// TODO: XMVectorRotateLeft template
// TODO: XMVectorRotateRight template
// TODO: XMVectorInsert template

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

#[inline]
pub fn XMVectorSplatConstant(IntConstant: i32, DivExponent: u32) -> XMVECTOR {
    assert!(IntConstant >= -16 && IntConstant <= 15);
    assert!(DivExponent < 32);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        use convert::XMConvertVectorIntToFloat;
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
