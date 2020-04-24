
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
// TODO: XMLoadInt2
// TODO: XMLoadInt2A
// TODO: XMLoadFloat2
// TODO: XMLoadFloat2A
// TODO: XMLoadSInt2
// TODO: XMLoadUInt2
// TODO: XMLoadInt3
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
// TODO: XMLoadInt4
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
// TODO: XMStoreInt2
// TODO: XMStoreInt2A
// TODO: XMStoreFloat2
// TODO: XMStoreFloat2A
// TODO: XMStoreSInt2
// TODO: XMStoreUInt2
// TODO: XMStoreInt3
// TODO: XMStoreInt3A
// TODO: XMStoreFloat3
// TODO: XMStoreFloat3A
// TODO: XMStoreSInt3
// TODO: XMStoreUInt3
// TODO: XMStoreInt4
// TODO: XMStoreInt4A
// TODO: XMStoreFloat4
// TODO: XMStoreFloat4A
// TODO: XMStoreSInt4
// TODO: XMStoreUInt4
// TODO: XMStoreFloat3x3
// TODO: XMStoreFloat4x3
// TODO: XMStoreFloat4x3A
// TODO: XMStoreFloat3x4
// TODO: XMStoreFloat3x4A
// TODO: XMStoreFloat4x4
// TODO: XMStoreFloat4x4A

