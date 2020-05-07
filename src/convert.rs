
#[allow(unused_imports)]
use std::mem;

use crate::*;

/// Converts an XMVECTOR with int32_t components to an XMVECTOR with float components and applies a uniform bias.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertVectorIntToFloat>
#[inline]
pub fn XMConvertVectorIntToFloat(VInt: FXMVECTOR, DivExponent: u32) -> XMVECTOR {
    debug_assert!(DivExponent < 32);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let fScale = 1.0 / (1 >> DivExponent) as f32;
        let mut Result: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        for ElementIndex in 0..4 {
            let iTemp = VInt.vector4_u32[ElementIndex];
            Result.vector4_f32[ElementIndex] = iTemp as f32 * fScale;
        }
        return Result;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // Convert to floats
        let mut vResult = _mm_cvtepi32_ps(_mm_castps_si128(VInt));
        // Convert DivExponent into 1.0f/(1<<DivExponent)
        let uScale = 0x3F800000u32 - (DivExponent << 23);
        // Splat the scalar value
        let vScale = _mm_set1_epi32(uScale as i32);
        vResult = _mm_mul_ps(vResult, _mm_castsi128_ps(vScale));
        return vResult;
    }
}

/// Converts an XMVECTOR with float components to an XMVECTOR with int32_t components and applies a uniform bias.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertVectorFloatToInt>
#[inline]
pub fn XMConvertVectorFloatToInt(VFloat: FXMVECTOR, MulExponent: u32) -> XMVECTOR {
    debug_assert!(MulExponent < 32);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let fScale = (1u32 << MulExponent) as f32;
        let mut Result: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        for ElementIndex in 0..4 {
            let iResult: i32;
            let fTemp: f32 = VFloat.vector4_f32[ElementIndex] * fScale;
            if (fTemp <= -(65536.0 * 32768.0))
            {
                iResult = (-0x7FFFFFFF) - 1;
            }
            else if (fTemp > (65536.0 * 32768.0) - 128.0)
            {
                iResult = 0x7FFFFFFF;
            }
            else {
                iResult = (fTemp as i32);
            }
            Result.vector4_u32[ElementIndex] = (iResult as u32);
        }
        return Result;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut vResult: XMVECTOR = _mm_set_ps1((1u32 << MulExponent) as f32);
        vResult = _mm_mul_ps(vResult, VFloat);
        // In case of positive overflow, detect it
        let mut vOverflow: XMVECTOR = _mm_cmpgt_ps(vResult, *g_XMMaxInt);
        // Float to int conversion
        let vResulti: __m128i = _mm_cvttps_epi32(vResult);
        // If there was positive overflow, set to 0x7FFFFFFF
        vResult = _mm_and_ps(vOverflow, *g_XMAbsMask);
        vOverflow = _mm_andnot_ps(vOverflow, _mm_castsi128_ps(vResulti));
        vOverflow = _mm_or_ps(vOverflow, vResult);
        return vOverflow;
    }
}

// TODO: XMConvertVectorUIntToFloat

/// Converts an XMVECTOR with uint32_t components to an XMVECTOR with float components and applies a uniform bias.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertVectorUIntToFloat>
#[inline]
pub fn XMConvertVectorUIntToFloat(
    VUInt: FXMVECTOR,
    DivExponent: u32
) -> XMVECTOR
{
    debug_assert!(DivExponent  < 32);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let fScale = 1.0 / (1u32 << DivExponent) as f32;
        let mut Result: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        for ElementIndex in 0..4 {
            Result.vector4_f32[ElementIndex] = (VUInt.vector4_u32[ElementIndex] as f32) * fScale;
        }
        return Result;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // For the values that are higher than 0x7FFFFFFF, a fixup is needed
        // Determine which ones need the fix.
        let mut vMask: XMVECTOR = _mm_and_ps(VUInt, *g_XMNegativeZero);
        // Force all values positive
        let mut vResult: XMVECTOR = _mm_xor_ps(VUInt, vMask);
        // Convert to floats
        vResult = _mm_cvtepi32_ps(_mm_castps_si128(vResult));
        // Convert 0x80000000 -> 0xFFFFFFFF
        let mut iMask: __m128i = _mm_srai_epi32(_mm_castps_si128(vMask), 31);
        // For only the ones that are too big, add the fixup
        vMask = _mm_and_ps(_mm_castsi128_ps(iMask), *g_XMFixUnsigned);
        vResult = _mm_add_ps(vResult, vMask);
        // Convert DivExponent into 1.0f/(1<<DivExponent)
        let uScale: u32 = 0x3F800000u32 - (DivExponent << 23);
        // Splat
        iMask = _mm_set1_epi32(uScale as i32);
        vResult = _mm_mul_ps(vResult, _mm_castsi128_ps(iMask));
        return vResult;
    }
}


/// Converts an XMVECTOR with uint32_t components to an XMVECTOR with float components and applies a uniform bias.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMConvertVectorUIntToFloat>
#[inline]
pub fn XMConvertVectorFloatToUInt(
    VFloat: FXMVECTOR,
    MulExponent: u32
) -> XMVECTOR
{
    debug_assert!(MulExponent  < 32);

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let fScale = (1u32 << MulExponent) as f32;
        let mut Result: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        for ElementIndex in 0..4 {
            let uResult: u32;
            let fTemp: f32 = VFloat.vector4_f32[ElementIndex] * fScale;
            if (fTemp <= 0.0)
            {
                uResult = 0;
            }
            else if (fTemp >= (65536.0 * 65536.0))
            {
                uResult = 0xFFFFFFFFu32;
            }
            else {
                uResult = (fTemp as u32);
            }
            Result.vector4_u32[ElementIndex] = uResult;
        }
        return Result;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut vResult: XMVECTOR = _mm_set_ps1((1u32 << MulExponent) as f32);
        vResult = _mm_mul_ps(vResult, VFloat);
        // Clamp to >=0
        vResult = _mm_max_ps(vResult, *g_XMZero);
        // Any numbers that are too big, set to 0xFFFFFFFFU
        let vOverflow: XMVECTOR = _mm_cmpgt_ps(vResult, *g_XMMaxUInt);
        let mut vValue: XMVECTOR = *g_XMUnsignedFix;
        // Too large for a signed integer?
        let mut vMask: XMVECTOR = _mm_cmpge_ps(vResult, vValue);
        // Zero for number's lower than 0x80000000, 32768.0f*65536.0f otherwise
        vValue = _mm_and_ps(vValue, vMask);
        // Perform fixup only on numbers too large (Keeps low bit precision)
        vResult = _mm_sub_ps(vResult, vValue);
        let vResulti: __m128i = _mm_cvttps_epi32(vResult);
        // Convert from signed to unsigned pnly if greater than 0x80000000
        vMask = _mm_and_ps(vMask, *g_XMNegativeZero);
        vResult = _mm_xor_ps(_mm_castsi128_ps(vResulti), vMask);
        // On those that are too large, set to 0xFFFFFFFF
        vResult = _mm_or_ps(vResult, vOverflow);
        return vResult;
    }
}

// TODO: XMLoadInt
// TODO: XMLoadFloat

/// Loads data into the x and y components of an XMVECTOR.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadInt2>
#[inline]
pub fn XMLoadInt2(
    pSource: &[u32; 2],
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        V.vector4_u32[0] = pSource[0];
        V.vector4_u32[1] = pSource[1];
        V.vector4_u32[2] = 0;
        V.vector4_u32[3] = 0;
        return V;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        return _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
    }
}

#[test]
fn test_XMLoadInt2() {
    let a = XMLoadInt2(&[-1i32 as u32, 1 as u32]);
    let b = XMVectorSetInt(-1i32 as u32, 1 as u32, 0, 0);
    assert_eq!(-1, XMVectorGetIntX(a) as i32);
    assert_eq!( 1, XMVectorGetIntY(a) as i32);
    assert!(XMVector2EqualInt(a, b));

    let c = XMLoadInt2(&[1, 2]);
    let d = XMVectorSetInt(1, 2 as u32, 0, 0);
    assert_eq!(1, XMVectorGetIntX(c));
    assert_eq!(2, XMVectorGetIntY(d));
    assert!(XMVector2EqualInt(c, d));
}

// TODO: XMLoadInt2A

/// Loads data into the x, y, and z components of an XMVECTOR, without type checking.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadInt3>
#[inline]
pub fn XMLoadFloat2(
    pSource: &XMFLOAT2,
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        V.vector4_f32[0] = pSource.x;
        V.vector4_f32[1] = pSource.y;
        V.vector4_f32[2] = 0.0;
        V.vector4_f32[3] = 0.0;
        return V;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        return _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
    }
}

// TODO: XMLoadFloat2A
// TODO: XMLoadSInt2
// TODO: XMLoadUInt2

/// Loads data into the x, y, and z components of an XMVECTOR, without type checking.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadInt3>
#[inline]
pub fn XMLoadInt3(
    pSource: &[u32; 3],
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        V.vector4_u32[0] = pSource[0];
        V.vector4_u32[1] = pSource[1];
        V.vector4_u32[2] = pSource[2];
        V.vector4_u32[3] = 0;
        return V;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE4_INTRINSICS_)]
    unsafe {
        let xy: __m128 = _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
        let z: __m128 = _mm_load_ss(mem::transmute::<_, *const f32>(&pSource[2]));
        return _mm_insert_ps(xy, z, 0x20);
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_SSE4_INTRINSICS_)))]
    unsafe {
        let xy: __m128 = _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
        let z: __m128 = _mm_load_ss(mem::transmute::<_, *const f32>(&pSource[2]));
        return _mm_movelh_ps(xy, z);
    }
}

#[test]
fn test_XMLoadInt3() {
    let a = XMLoadInt3(&[-1i32 as u32, 0 as u32, 1 as u32]);
    let b = XMVectorSetInt(-1i32 as u32, 0 as u32, 1, 0);
    assert_eq!(-1, XMVectorGetIntX(a) as i32);
    assert_eq!( 0, XMVectorGetIntY(a) as i32);
    assert_eq!( 1, XMVectorGetIntZ(a) as i32);
    assert!(XMVector3EqualInt(a, b));

    let c = XMLoadInt3(&[1, 2, 3]);
    let d = XMVectorSetInt(1, 2 as u32, 3, 0);
    assert_eq!(1, XMVectorGetIntX(c));
    assert_eq!(2, XMVectorGetIntY(d));
    assert_eq!(3, XMVectorGetIntZ(d));
    assert!(XMVector3EqualInt(c, d));
}

// TODO: XMLoadInt3A

/// Loads an XMFLOAT3 into an XMVECTOR.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadFloat3>
#[inline]
pub fn XMLoadFloat3(
    pSource: &XMFLOAT3,
) -> XMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        V.vector4_f32[0] = pSource.x;
        V.vector4_f32[1] = pSource.y;
        V.vector4_f32[2] = pSource.z;
        V.vector4_f32[3] = 0.0;
        return V;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE4_INTRINSICS_)]
    unsafe {
        let xy: __m128 = _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
        let z: __m128 = _mm_load_ss(&pSource.z);
        return _mm_insert_ps(xy, z, 0x20);
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_SSE4_INTRINSICS_)))]
    unsafe {
        let xy: __m128 = _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
        let z: __m128 = _mm_load_ss(&pSource.z);
        return _mm_movelh_ps(xy, z);
    }
}

#[test]
fn test_XMLoadFloat3() {
    let a = XMLoadFloat3(&XMFLOAT3 { x: 1.0, y: 2.0, z: 3.0 });
    assert_eq!(1.0, XMVectorGetX(a));
    assert_eq!(2.0, XMVectorGetY(a));
    assert_eq!(3.0, XMVectorGetZ(a));
    assert_eq!(0.0, XMVectorGetW(a));
}

// TODO: XMLoadFloat3A

// TODO: XMLoadSInt3

// /// Loads signed integer data into the x, y, and z components of an XMVECTOR.
// ///
// /// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadSInt3>
// #[inline]
// pub fn XMLoadSInt3(
//     pSource: &XMINT3,
// ) -> FXMVECTOR
// {
//     #[cfg(_XM_NO_INTRINSICS_)]
//     unsafe {
//         let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
//         V.vector4_f32[0] = (pSource.x as f32);
//         V.vector4_f32[1] = (pSource.y as f32);
//         V.vector4_f32[2] = (pSource.z as f32);
//         V.vector4_f32[3] = 0.0;
//         return V;
//     }

//     #[cfg(_XM_ARM_NEON_INTRINSICS_)]
//     {
//         unimplemented!()
//     }

//     #[cfg(_XM_SSE_INTRINSICS_)]
//     unsafe {
//         let xy: __m128 = _mm_castpd_ps(_mm_load_sd(mem::transmute::<_, *const f64>(pSource)));
//         let z: __m128 = _mm_load_ss(mem::transmute::<_, *const f32>(&pSource.z));
//         let V: __m128 = _mm_movelh_ps(xy, z);
//         return _mm_cvtepi32_ps(_mm_castps_si128(V));
//     }
// }

// #[test]
// fn test_XMLoadSInt3() {
//     let a = XMLoadSInt3(&XMINT3 { x: 1, y: 2, z: 3 });
//     assert_eq!(1.0, XMVectorGetX(a));
//     assert_eq!(2.0, XMVectorGetY(a));
//     assert_eq!(3.0, XMVectorGetZ(a));
//     assert_eq!(0.0, XMVectorGetW(a));

//     let a = XMLoadSInt3(&XMINT3 { x: -1, y: -2, z: -3 });
//     assert_eq!(-1.0, XMVectorGetX(a));
//     assert_eq!(-2.0, XMVectorGetY(a));
//     assert_eq!(-3.0, XMVectorGetZ(a));
//     assert_eq!(0.0, XMVectorGetW(a));
// }


// TODO: XMLoadUInt3

/// Loads data into an XMVECTOR, without type checking.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadInt4>
#[inline]
pub fn XMLoadInt4(
    pSource: &[u32; 4],
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        V.vector4_u32[0] = pSource[0];
        V.vector4_u32[1] = pSource[1];
        V.vector4_u32[2] = pSource[2];
        V.vector4_u32[3] = pSource[3];
        return V;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(all(_XM_SSE_INTRINSICS_))]
    unsafe {
        let V: __m128i = _mm_loadu_si128(mem::transmute::<_, *const __m128i>(pSource));
        return _mm_castsi128_ps(V);
    }
}

#[test]
fn test_XMLoadInt4() {
    let a = XMLoadInt4(&[-1i32 as u32, 0 as u32, 1 as u32, 2 as u32]);
    let b = XMVectorSetInt(-1i32 as u32, 0 as u32, 1, 2);
    assert_eq!(-1, XMVectorGetIntX(a) as i32);
    assert_eq!( 0, XMVectorGetIntY(a) as i32);
    assert_eq!( 1, XMVectorGetIntZ(a) as i32);
    assert_eq!( 2, XMVectorGetIntW(a) as i32);
    assert!(XMVector4EqualInt(a, b));

    let c = XMLoadInt4(&[1, 2, 3, 4]);
    let d = XMVectorSetInt(1, 2 as u32, 3, 4);
    assert_eq!(1, XMVectorGetIntX(c));
    assert_eq!(2, XMVectorGetIntY(d));
    assert_eq!(3, XMVectorGetIntZ(d));
    assert_eq!(4, XMVectorGetIntW(d));
    assert!(XMVector4EqualInt(c, d));
}

// TODO: XMLoadInt4A

/// Loads an XMFLOAT4 into an XMVECTOR.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadFloat4>
#[inline]
pub fn XMLoadFloat4(
    pSource: &XMFLOAT4,
) -> XMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut V: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        V.vector4_f32[0] = pSource.x;
        V.vector4_f32[1] = pSource.y;
        V.vector4_f32[2] = pSource.z;
        V.vector4_f32[3] = pSource.w;
        return V;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        return _mm_loadu_ps(&pSource.x);
    }
}

#[test]
fn test_XMLoadFloat4() {
    let a = XMLoadFloat4(&XMFLOAT4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 });
    assert_eq!(1.0, XMVectorGetX(a));
    assert_eq!(2.0, XMVectorGetY(a));
    assert_eq!(3.0, XMVectorGetZ(a));
    assert_eq!(4.0, XMVectorGetW(a));
}

// TODO: XMLoadFloat4A
// TODO: XMLoadSInt4
// TODO: XMLoadUInt4

/// Loads an XMFLOAT3X3 into an MATRIX.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadFloat3x3>
#[inline]
pub fn XMLoadFloat3x3(
    pSource: &XMFLOAT3X3,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX  = mem::MaybeUninit::uninit().assume_init();
        M.r[0].vector4_f32[0] = pSource.m[0][0];
        M.r[0].vector4_f32[1] = pSource.m[0][1];
        M.r[0].vector4_f32[2] = pSource.m[0][2];
        M.r[0].vector4_f32[3] = 0.0;

        M.r[1].vector4_f32[0] = pSource.m[1][0];
        M.r[1].vector4_f32[1] = pSource.m[1][1];
        M.r[1].vector4_f32[2] = pSource.m[1][2];
        M.r[1].vector4_f32[3] = 0.0;

        M.r[2].vector4_f32[0] = pSource.m[2][0];
        M.r[2].vector4_f32[1] = pSource.m[2][1];
        M.r[2].vector4_f32[2] = pSource.m[2][2];
        M.r[2].vector4_f32[3] = 0.0;

        M.r[3].vector4_f32[0] = 0.0;
        M.r[3].vector4_f32[1] = 0.0;
        M.r[3].vector4_f32[2] = 0.0;
        M.r[3].vector4_f32[3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let Z: __m128 = _mm_setzero_ps();

        let V1: __m128 = _mm_loadu_ps(&pSource.m[0][0]);
        let V2: __m128 = _mm_loadu_ps(&pSource.m[1][1]);
        let V3: __m128 = _mm_load_ss(&pSource.m[2][2]);

        let T1: __m128 = _mm_unpackhi_ps(V1, Z);
        let T2: __m128 = _mm_unpacklo_ps(V2, Z);
        let T3: __m128 = _mm_shuffle_ps(V3, T2, _MM_SHUFFLE(0, 1, 0, 0));
        let T4: __m128 = _mm_movehl_ps(T2, T3);
        let T5: __m128 = _mm_movehl_ps(Z, T1);

        let mut M: XMMATRIX  = mem::MaybeUninit::uninit().assume_init();
        M.r[0] = _mm_movelh_ps(V1, T1);
        M.r[1] = _mm_add_ps(T4, T5);
        M.r[2] = _mm_shuffle_ps(V2, V3, _MM_SHUFFLE(1, 0, 3, 2));
        M.r[3] = *g_XMIdentityR3;
        return M;
    }
}

// TODO: XMLoadFloat4x3
// TODO: XMLoadFloat4x3A
// TODO: XMLoadFloat3x4
// TODO: XMLoadFloat3x4A

/// Loads an XMFLOAT4X4 into an MATRIX.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMLoadFloat4x4>
#[inline]
pub fn XMLoadFloat4x4(
    pSource: &XMFLOAT4X4,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX  = mem::MaybeUninit::uninit().assume_init();
        M.r[0].vector4_f32[0] = pSource.m[0][0];
        M.r[0].vector4_f32[1] = pSource.m[0][1];
        M.r[0].vector4_f32[2] = pSource.m[0][2];
        M.r[0].vector4_f32[3] = pSource.m[0][3];

        M.r[1].vector4_f32[0] = pSource.m[1][0];
        M.r[1].vector4_f32[1] = pSource.m[1][1];
        M.r[1].vector4_f32[2] = pSource.m[1][2];
        M.r[1].vector4_f32[3] = pSource.m[1][3];

        M.r[2].vector4_f32[0] = pSource.m[2][0];
        M.r[2].vector4_f32[1] = pSource.m[2][1];
        M.r[2].vector4_f32[2] = pSource.m[2][2];
        M.r[2].vector4_f32[3] = pSource.m[2][3];

        M.r[3].vector4_f32[0] = pSource.m[3][0];
        M.r[3].vector4_f32[1] = pSource.m[3][1];
        M.r[3].vector4_f32[2] = pSource.m[3][2];
        M.r[3].vector4_f32[3] = pSource.m[3][3];
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX  = mem::MaybeUninit::uninit().assume_init();
        M.r[0] = _mm_loadu_ps(&pSource.m[0][0]); // _11
        M.r[1] = _mm_loadu_ps(&pSource.m[1][0]); // _21
        M.r[2] = _mm_loadu_ps(&pSource.m[2][0]); // _31
        M.r[3] = _mm_loadu_ps(&pSource.m[3][0]); // _41
        return M;
    }
}

// TODO: XMLoadFloat4x4A
// TODO: XMStoreInt
// TODO: XMStoreFloat

/// Stores an XMVECTOR in a float.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreFloat>
#[inline]
pub fn XMStoreFloat(
    pDestination: &mut f32,
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    {
        *pDestination = XMVectorGetX(V);
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(all(_XM_SSE_INTRINSICS_))]
    unsafe {
        _mm_store_ss(pDestination, V);
    }
}

/// Stores an XMVECTOR in a 2-element uint32_t array.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreInt2>
#[inline]
pub fn XMStoreInt2(
    pDestination: &mut [u32; 2],
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination[0] = V.vector4_u32[0];
        pDestination[1] = V.vector4_u32[1];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(all(_XM_SSE_INTRINSICS_))]
    unsafe {
        _mm_store_sd(mem::transmute::<_, *mut f64>(pDestination), _mm_castps_pd(V));
    }
}

#[test]
fn test_XMStoreInt2() {
    let mut a: [u32; 2] = [0, 0];
    XMStoreInt2(&mut a, XMVectorSetInt(1, 2, 0, 0));
    assert_eq!(1, a[0]);
    assert_eq!(2, a[1]);

    let mut b: [u32; 2] = [-1i32 as u32, 1];
    XMStoreInt2(&mut b, XMVectorSetInt(-1i32 as u32, 1, 0, 0));
    assert_eq!(-1, b[0] as i32);
    assert_eq!( 1, b[1] as i32);
}

// TODO: XMStoreInt2A

/// Stores an XMVECTOR in an XMFLOAT2.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreFloat2>
#[inline]
pub fn XMStoreFloat2(
    pDestination: &mut XMFLOAT2,
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination.x = V.vector4_f32[0];
        pDestination.y = V.vector4_f32[1];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(all(_XM_SSE_INTRINSICS_))]
    unsafe {
        let pDestination: *mut XMFLOAT2 = mem::transmute(pDestination);
        _mm_store_sd(mem::transmute::<_, *mut f64>(pDestination), _mm_castps_pd(V));
    }
}

#[test]
fn test_XMStoreFloat2() {
    let mut a = XMFLOAT2 { x: 0.0, y: 0.0 };
    XMStoreFloat2(&mut a, XMVectorSet(1.0, 2.0, 0.0, 0.0));
    assert_eq!(1.0, a.x);
    assert_eq!(2.0, a.y);
}

// TODO: XMStoreFloat2A
// TODO: XMStoreSInt2
// TODO: XMStoreUInt2

/// Stores an XMVECTOR in a 3-element uint32_t array.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreInt3>
#[inline]
pub fn XMStoreInt3(
    pDestination: &mut [u32; 3],
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination[0] = V.vector4_u32[0];
        pDestination[1] = V.vector4_u32[1];
        pDestination[2] = V.vector4_u32[2];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(all(_XM_SSE_INTRINSICS_))]
    unsafe {
        _mm_store_sd(mem::transmute::<_, *mut f64>(&mut *pDestination), _mm_castps_pd(V));
        let z: __m128 = XM_PERMUTE_PS!(V, _MM_SHUFFLE(2, 2, 2, 2));
        _mm_store_ss(mem::transmute::<_, *mut f32>(&mut pDestination[2]), z);
    }
}

#[test]
fn test_XMStoreInt3() {
    let mut a: [u32; 3] = [0, 0, 0];
    XMStoreInt3(&mut a, XMVectorSetInt(1, 2, 3, 0));
    assert_eq!(1, a[0]);
    assert_eq!(2, a[1]);
    assert_eq!(3, a[2]);

    let mut b: [u32; 3] = [-1i32 as u32, 1, 2];
    XMStoreInt3(&mut b, XMVectorSetInt(-1i32 as u32, 1, 2, 0));
    assert_eq!(-1, b[0] as i32);
    assert_eq!( 1, b[1] as i32);
    assert_eq!( 2, b[2] as i32);
}

// TODO: XMStoreInt3A

/// Stores an XMVECTOR in an XMFLOAT3.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreFloat3>
#[inline]
pub fn XMStoreFloat3(
    pDestination: &mut XMFLOAT3,
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination.x = V.vector4_f32[0];
        pDestination.y = V.vector4_f32[1];
        pDestination.z = V.vector4_f32[2];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE4_INTRINSICS_)]
    unsafe {
        *mem::transmute::<_, *mut i32>(&mut pDestination.x) = _mm_extract_ps(V, 0);
        *mem::transmute::<_, *mut i32>(&mut pDestination.y) = _mm_extract_ps(V, 1);
        *mem::transmute::<_, *mut i32>(&mut pDestination.z) = _mm_extract_ps(V, 2);
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_SSE4_INTRINSICS_)))]
    unsafe {
        let pDestination: *mut XMFLOAT3 = mem::transmute(pDestination);
        _mm_store_sd(mem::transmute::<_, *mut f64>(pDestination), _mm_castps_pd(V));
        let z: __m128 = XM_PERMUTE_PS!(V, _MM_SHUFFLE(2, 2, 2, 2));
        _mm_store_ss(&mut (*pDestination).z, z);
    }
}

#[test]
fn test_XMStoreFloat3() {
    let mut a = XMFLOAT3 { x: 0.0, y: 0.0, z: 0.0 };
    XMStoreFloat3(&mut a, XMVectorSet(1.0, 2.0, 3.0, 0.0));
    assert_eq!(1.0, a.x);
    assert_eq!(2.0, a.y);
    assert_eq!(3.0, a.z);
}

// TODO: XMStoreFloat3A
// TODO: XMStoreSInt3
// TODO: XMStoreUInt3

/// Stores an XMVECTOR in a 4-element uint32_t array.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreInt4>
#[inline]
pub fn XMStoreInt4(
    pDestination: &mut [u32; 4],
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination[0] = V.vector4_u32[0];
        pDestination[1] = V.vector4_u32[1];
        pDestination[2] = V.vector4_u32[2];
        pDestination[3] = V.vector4_u32[3];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(all(_XM_SSE_INTRINSICS_))]
    unsafe {
        _mm_storeu_si128(mem::transmute::<_, *mut __m128i>(pDestination), _mm_castps_si128(V));
    }
}

#[test]
fn test_XMStoreInt4() {
    let mut a: [u32; 4] = [0, 0, 0, 0];
    XMStoreInt4(&mut a, XMVectorSetInt(1, 2, 3, 4));
    assert_eq!(1, a[0]);
    assert_eq!(2, a[1]);
    assert_eq!(3, a[2]);
    assert_eq!(4, a[3]);

    let mut b: [u32; 4] = [-1i32 as u32, 1, 2, 3];
    XMStoreInt4(&mut b, XMVectorSetInt(-1i32 as u32, 1, 2, 3));
    assert_eq!(-1, b[0] as i32);
    assert_eq!( 1, b[1] as i32);
    assert_eq!( 2, b[2] as i32);
    assert_eq!( 3, b[3] as i32);
}

// TODO: XMStoreInt4A

/// Stores an XMVECTOR in an XMFLOAT4.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreFloat4>
#[inline]
pub fn XMStoreFloat4(
    pDestination: &mut XMFLOAT4,
    V: FXMVECTOR,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination.x = V.vector4_f32[0];
        pDestination.y = V.vector4_f32[1];
        pDestination.z = V.vector4_f32[2];
        pDestination.w = V.vector4_f32[3];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        _mm_storeu_ps(&mut pDestination.x, V);
    }
}

#[test]
fn test_XMStoreFloat4() {
    let mut a = XMFLOAT4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    XMStoreFloat4(&mut a, XMVectorSet(1.0, 2.0, 3.0, 4.0));
    assert_eq!(1.0, a.x);
    assert_eq!(2.0, a.y);
    assert_eq!(3.0, a.z);
    assert_eq!(4.0, a.w);
}

// TODO: XMStoreFloat4A
// TODO: XMStoreSInt4
// TODO: XMStoreUInt4

/// Stores an XMMATRIX in an XMFLOAT3X3.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreFloat3x3>
#[inline]
pub fn XMStoreFloat3x3(
    pDestination: &mut XMFLOAT3X3,
    M: FXMMATRIX,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination.m[0][0] = M.r[0].vector4_f32[0];
        pDestination.m[0][1] = M.r[0].vector4_f32[1];
        pDestination.m[0][2] = M.r[0].vector4_f32[2];

        pDestination.m[1][0] = M.r[1].vector4_f32[0];
        pDestination.m[1][1] = M.r[1].vector4_f32[1];
        pDestination.m[1][2] = M.r[1].vector4_f32[2];

        pDestination.m[2][0] = M.r[2].vector4_f32[0];
        pDestination.m[2][1] = M.r[2].vector4_f32[1];
        pDestination.m[2][2] = M.r[2].vector4_f32[2];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut vTemp1: XMVECTOR = M.r[0];
        let mut vTemp2: XMVECTOR = M.r[1];
        let mut vTemp3: XMVECTOR = M.r[2];
        let vWork: XMVECTOR = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(0, 0, 2, 2));
        vTemp1 = _mm_shuffle_ps(vTemp1, vWork, _MM_SHUFFLE(2, 0, 1, 0));
        _mm_storeu_ps(&mut pDestination.m[0][0], vTemp1);
        vTemp2 = _mm_shuffle_ps(vTemp2, vTemp3, _MM_SHUFFLE(1, 0, 2, 1));
        _mm_storeu_ps(&mut pDestination.m[1][1], vTemp2);
        vTemp3 = XM_PERMUTE_PS!(vTemp3, _MM_SHUFFLE(2, 2, 2, 2));
        _mm_store_ss(&mut pDestination.m[2][2], vTemp3);
    }
}

// TODO: XMStoreFloat4x3
// TODO: XMStoreFloat4x3A
// TODO: XMStoreFloat3x4
// TODO: XMStoreFloat3x4A

/// Stores an XMMATRIX in an XMFLOAT4X4.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMStoreFloat4x4>
#[inline]
pub fn XMStoreFloat4x4(
    pDestination: &mut XMFLOAT4X4,
    M: FXMMATRIX,
)
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        pDestination.m[0][0] = M.r[0].vector4_f32[0];
        pDestination.m[0][1] = M.r[0].vector4_f32[1];
        pDestination.m[0][2] = M.r[0].vector4_f32[2];
        pDestination.m[0][3] = M.r[0].vector4_f32[3];

        pDestination.m[1][0] = M.r[1].vector4_f32[0];
        pDestination.m[1][1] = M.r[1].vector4_f32[1];
        pDestination.m[1][2] = M.r[1].vector4_f32[2];
        pDestination.m[1][3] = M.r[1].vector4_f32[3];

        pDestination.m[2][0] = M.r[2].vector4_f32[0];
        pDestination.m[2][1] = M.r[2].vector4_f32[1];
        pDestination.m[2][2] = M.r[2].vector4_f32[2];
        pDestination.m[2][3] = M.r[2].vector4_f32[3];

        pDestination.m[3][0] = M.r[3].vector4_f32[0];
        pDestination.m[3][1] = M.r[3].vector4_f32[1];
        pDestination.m[3][2] = M.r[3].vector4_f32[2];
        pDestination.m[3][3] = M.r[3].vector4_f32[3];
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        _mm_storeu_ps(&mut pDestination.m[0][0], M.r[0]); // _11
        _mm_storeu_ps(&mut pDestination.m[1][0], M.r[1]); // _21
        _mm_storeu_ps(&mut pDestination.m[2][0], M.r[2]); // _31
        _mm_storeu_ps(&mut pDestination.m[3][0], M.r[3]); // _41
    }
}

// TODO: XMStoreFloat4x4A

