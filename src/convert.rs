
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
// TODO: XMConvertVectorFloatToUInt
// TODO: XMLoadInt