#[allow(unused_imports)]
use std::mem;

use crate::*;

/// Converts a half-precision floating-point value to a single-precision floating-point value.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxpackedvector/nf-directxpackedvector-XMConvertHalfToFloat>
#[inline]
pub fn XMConvertHalfToFloat(Value: HALF) -> f32 {
    #[cfg(any(_XM_NO_INTRINSICS_, not(_XM_F16C_INTRINSICS_)))]
    {
        let mut Mantissa = (Value as u32 & 0x03FF);

        let mut Exponent: u32 = (Value as u32 & 0x7C00);
        if (Exponent == 0x7C00) // INF/NAN
        {
            Exponent = 0x8f;
        }
        else if (Exponent != 0)  // The value is normalized
        {
            Exponent = (((Value as i32) >> 10) & 0x1F) as u32;
        }
        else if (Mantissa != 0)     // The value is denormalized
        {
            // Normalize the value in the resulting float
            Exponent = 1;

            loop
            {
                Exponent = Exponent.wrapping_sub(1);
                Mantissa <<= 1;
                if ((Mantissa & 0x0400) == 0) {
                    break;
                }
            }

            Mantissa &= 0x03FF;
        }
        else                        // The value is zero
        {
            Exponent = ((-112_i32) as u32);
        }

        let Result: u32 =
            ((Value as u32 & 0x8000) << 16)                 // Sign
            | ((Exponent.wrapping_add(112)) << 23)                      // Exponent
            | (Mantissa << 13);                             // Mantissa

        return f32::from_bits(Result);
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_F16C_INTRINSICS_)]
    unsafe {
        let V1: __m128i = _mm_cvtsi32_si128(Value as i32);
        let V2: __m128 = _mm_cvtph_ps(V1);
        return _mm_cvtss_f32(V2);
    }
}

/// Converts a single-precision floating-point value to a half-precision floating-point value.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxpackedvector/nf-directxpackedvector-XMConvertFloatToHalf>
#[inline]
pub fn XMConvertFloatToHalf(Value: f32) -> HALF {
    #[cfg(any(_XM_NO_INTRINSICS_, not(_XM_F16C_INTRINSICS_)))]
    unsafe {
        let mut Result: u32;

        let mut IValue = *mem::transmute::<_, *const u32>(&Value);
        let Sign: u32 = (IValue & 0x80000000u32) >> 16u32;
        IValue = IValue & 0x7FFFFFFFu32;      // Hack off the sign
        if (IValue >= 0x47800000 /*e+16*/)
        {
            // The number is too large to be represented as a half. Return infinity or NaN
            Result = 0x7C00u32 | (if (IValue > 0x7F800000) { (0x200 | ((IValue >> 13u32) & 0x3FFu32)) } else { 0u32 });
        }
        else if (IValue <= 0x33000000u32 /*e-25*/)
        {
            Result = 0;
        }
        else if (IValue < 0x38800000u32 /*e-14*/)
        {
            // The number is too small to be represented as a normalized half.
            // Convert it to a denormalized value.
            let Shift: u32 = 125u32 - (IValue >> 23u32);
            IValue = 0x800000u32 | (IValue & 0x7FFFFFu32);
            Result = IValue >> (Shift.wrapping_add(1));
            let s: u32 = if (IValue & ((1u32 << Shift).wrapping_sub(1))) != 0 { 1 } else { 0 };
            Result += (Result | s) & ((IValue >> Shift) & 1u32);
        }
        else
        {
            // Rebias the exponent to represent the value as a normalized half.
            IValue = IValue.wrapping_add(0xC8000000u32);
            Result = ((IValue + 0x0FFFu32 + ((IValue >> 13u32) & 1u32)) >> 13u32) & 0x7FFFu32;
        }
        return (Result | Sign) as u16;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_F16C_INTRINSICS_)]
    unsafe {
        let V1: __m128 = _mm_set_ss(Value);
        let V2: __m128i = _mm_cvtps_ph(V1, _MM_FROUND_TO_NEAREST_INT);
        return _mm_extract_epi16(V2, 0) as u16
    }
}

#[test]
fn test_half_conversion() {
    for v in [-1.0, 0.0, 1.0].iter().cloned() {
        let h = XMConvertFloatToHalf(v);
        assert_eq!(v, XMConvertHalfToFloat(h));
    }
}