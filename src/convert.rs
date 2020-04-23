
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

// TODO: LoadInt
// TODO: LoadFloat

// TODO: LoadFloat2
// TODO: LoadSInt2
// TODO: LoadUInt2

// TODO: LoadFloat3
// TODO: LoadSInt3
// TODO: LoadUInt3

// TODO: LoadFloat4
// TODO: LoadSInt4
// TODO: LoadUInt4

// TODO: XMLoadFloat3x3
// TODO: XMLoadFloat4x3
// TODO: XMLoadFloat3x4
// TODO: XMLoadFloat4x4

// TODO: XMStoreFloat2