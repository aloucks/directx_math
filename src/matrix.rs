use crate::*;
use std::mem;

/// Tests whether any of the elements of a matrix are NaN.
///
/// ## Parameters
///
/// `M` Matrix to test.
///
/// ## Return value
///
/// Returns `true` if any element of `M` is `NaN`, and `false` otherwise.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixIsNaN>
#[inline]
pub fn XMMatrixIsNaN(
    M: FXMMATRIX,
) -> bool
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let pWork: &[u32; 16] = mem::transmute(&M.m[0][0]);
        for mut uTest in pWork.iter().cloned() {
            // Remove sign
            uTest &= 0x7FFFFFFFu32;
            // NaN is 0x7F800001 through 0x7FFFFFFF inclusive
            uTest = uTest.wrapping_sub(0x7F800001u32);
            if (uTest < 0x007FFFFFu32) {
                return true;
            }
        }
        false
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // Load in registers
        let mut vX: XMVECTOR = M.r[0];
        let mut vY: XMVECTOR = M.r[1];
        let mut vZ: XMVECTOR = M.r[2];
        let mut vW: XMVECTOR = M.r[3];
        // Test themselves to check for NaN
        vX = _mm_cmpneq_ps(vX, vX);
        vY = _mm_cmpneq_ps(vY, vY);
        vZ = _mm_cmpneq_ps(vZ, vZ);
        vW = _mm_cmpneq_ps(vW, vW);
        // Or all the results
        vX = _mm_or_ps(vX, vZ);
        vY = _mm_or_ps(vY, vW);
        vX = _mm_or_ps(vX, vY);
        // If any tested true, return true
        return (_mm_movemask_ps(vX) != 0);
    }
}

#[test]
fn test_XMMatrixIsNan() {
    let m = XMMATRIX {
        r: [*g_XMZero, *g_XMZero, *g_XMZero, *g_XMZero],
    };
    assert_eq!(false, XMMatrixIsNaN(m));

    let m = XMMATRIX {
        r: [*g_XMInfinity, *g_XMInfinity, *g_XMInfinity, *g_XMInfinity],
    };
    assert_eq!(false, XMMatrixIsNaN(m));

    let m = XMMATRIX {
        r: [*g_XMZero, *g_XMZero, *g_XMZero, *g_XMNegQNaN],
    };
    assert_eq!(true, XMMatrixIsNaN(m));

    let m = XMMATRIX {
        r: [XMVectorReplicate(std::f32::NAN), *g_XMZero, *g_XMZero, *g_XMZero],
    };
    assert_eq!(true, XMMatrixIsNaN(m));
}

/// Tests whether any of the elements of a matrix are infinite.
///
/// ## Parameters
///
/// `M` Matrix to test.
///
/// ## Return value
///
/// Returns `true` if any element of `M` is either positive or negative infinity, and `false` otherwise.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixIsInfinite>
#[inline]
pub fn XMMatrixIsInfinite(
    M: FXMMATRIX,
) -> bool
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let pWork: &[u32; 16] = mem::transmute(&M.m[0][0]);
        for mut uTest in pWork.iter().cloned() {
            // Remove sign
            uTest &= 0x7FFFFFFFu32;
            // INF is 0x7F800000
            if (uTest == 0x7F800000u32) {
                return true;
            }
        }
        false
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // Mask off the sign bits
        let mut vTemp1: XMVECTOR = _mm_and_ps(M.r[0], *g_XMAbsMask);
        let mut vTemp2: XMVECTOR = _mm_and_ps(M.r[1], *g_XMAbsMask);
        let mut vTemp3: XMVECTOR = _mm_and_ps(M.r[2], *g_XMAbsMask);
        let mut vTemp4: XMVECTOR = _mm_and_ps(M.r[3], *g_XMAbsMask);
        // Compare to infinity
        vTemp1 = _mm_cmpeq_ps(vTemp1, *g_XMInfinity);
        vTemp2 = _mm_cmpeq_ps(vTemp2, *g_XMInfinity);
        vTemp3 = _mm_cmpeq_ps(vTemp3, *g_XMInfinity);
        vTemp4 = _mm_cmpeq_ps(vTemp4, *g_XMInfinity);
        // Or the answers together
        vTemp1 = _mm_or_ps(vTemp1, vTemp2);
        vTemp3 = _mm_or_ps(vTemp3, vTemp4);
        vTemp1 = _mm_or_ps(vTemp1, vTemp3);
        // If any are infinity, the signs are true.
        return (_mm_movemask_ps(vTemp1) != 0);
    }
}

#[test]
fn test_XMMatrixIsInfinite() {
    let m = XMMATRIX {
        r: [*g_XMZero, *g_XMZero, *g_XMZero, *g_XMZero],
    };
    assert_eq!(false, XMMatrixIsInfinite(m));

    let m = XMMATRIX {
        r: [*g_XMQNaN, *g_XMQNaN, *g_XMQNaN, *g_XMQNaN],
    };
    assert_eq!(false, XMMatrixIsInfinite(m));

    let m = XMMATRIX {
        r: [*g_XMZero, *g_XMZero, *g_XMZero, *g_XMNegInfinity],
    };
    assert_eq!(true, XMMatrixIsInfinite(m));

    let m = XMMATRIX {
        r: [*g_XMZero, *g_XMZero, *g_XMZero, *g_XMInfinity],
    };
    assert_eq!(true, XMMatrixIsInfinite(m));
}

/// Tests whether a matrix is the identity matrix.
///
/// ## Parameters
///
/// `M` Matrix to test.
///
/// ## Return value
///
/// Returns `true` if `M` is the identity matrix, and `false` otherwise.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixIsIdentity>
#[inline]
pub fn XMMatrixIsIdentity(
    M: FXMMATRIX,
) -> bool
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        // Use the integer pipeline to reduce branching to a minimum
        let pWork: &[u32; 16] = mem::transmute(&M.m[0][0]);
        // Convert 1.0f to zero and or them together
        let mut uOne: u32 = pWork[0] ^ 0x3F800000u32;
        // Or all the 0.0f entries together
        let mut uZero: u32 = pWork[1];
        uZero |= pWork[2];
        uZero |= pWork[3];
        // 2nd row
        uZero |= pWork[4];
        uOne |= pWork[5] ^ 0x3F800000u32;
        uZero |= pWork[6];
        uZero |= pWork[7];
        // 3rd row
        uZero |= pWork[8];
        uZero |= pWork[9];
        uOne |= pWork[10] ^ 0x3F800000u32;
        uZero |= pWork[11];
        // 4th row
        uZero |= pWork[12];
        uZero |= pWork[13];
        uZero |= pWork[14];
        uOne |= pWork[15] ^ 0x3F800000u32;
        // If all zero entries are zero, the uZero==0
        uZero &= 0x7FFFFFFF;    // Allow -0.0f
        // If all 1.0f entries are 1.0f, then uOne==0
        uOne |= uZero;
        return (uOne == 0);
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut vTemp1: XMVECTOR = _mm_cmpeq_ps(M.r[0], *g_XMIdentityR0);
        let vTemp2: XMVECTOR = _mm_cmpeq_ps(M.r[1], *g_XMIdentityR1);
        let mut vTemp3: XMVECTOR = _mm_cmpeq_ps(M.r[2], *g_XMIdentityR2);
        let vTemp4: XMVECTOR = _mm_cmpeq_ps(M.r[3], *g_XMIdentityR3);
        vTemp1 = _mm_and_ps(vTemp1, vTemp2);
        vTemp3 = _mm_and_ps(vTemp3, vTemp4);
        vTemp1 = _mm_and_ps(vTemp1, vTemp3);
        return (_mm_movemask_ps(vTemp1) == 0x0f);
    }
}

/// Computes the product of two matrices.
///
/// ## Parameters
///
/// `M1` First matrix to multiply.
///
/// `M2` Second matrix to multiply.
///
/// ## Return value
///
/// Returns the product of `M1` and `M2`.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixMultiply>
#[inline]
pub fn XMMatrixMultiply(
    M1: FXMMATRIX,
    M2: CXMMATRIX,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut mResult: XMMATRIX = crate::undefined();
        // Cache the invariants in registers
        let mut x: f32 = M1.m[0][0];
        let mut y: f32 = M1.m[0][1];
        let mut z: f32 = M1.m[0][2];
        let mut w: f32 = M1.m[0][3];
        // Perform the operation on the first row
        mResult.m[0][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
        mResult.m[0][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
        mResult.m[0][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
        mResult.m[0][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
        // Repeat for all the other rows
        x = M1.m[1][0];
        y = M1.m[1][1];
        z = M1.m[1][2];
        w = M1.m[1][3];
        mResult.m[1][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
        mResult.m[1][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
        mResult.m[1][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
        mResult.m[1][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
        x = M1.m[2][0];
        y = M1.m[2][1];
        z = M1.m[2][2];
        w = M1.m[2][3];
        mResult.m[2][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
        mResult.m[2][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
        mResult.m[2][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
        mResult.m[2][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
        x = M1.m[3][0];
        y = M1.m[3][1];
        z = M1.m[3][2];
        w = M1.m[3][3];
        mResult.m[3][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
        mResult.m[3][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
        mResult.m[3][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
        mResult.m[3][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
        return mResult;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_AVX2_INTRINSICS_)]
    unsafe {
        let mut t0: __m256 = _mm256_castps128_ps256(M1.r[0]);
        t0 = _mm256_insertf128_ps(t0, M1.r[1], 1);
        let mut t1: __m256 = _mm256_castps128_ps256(M1.r[2]);
        t1 = _mm256_insertf128_ps(t1, M1.r[3], 1);

        let mut u0: __m256 = _mm256_castps128_ps256(M2.r[0]);
        u0 = _mm256_insertf128_ps(u0, M2.r[1], 1);
        let mut u1: __m256 = _mm256_castps128_ps256(M2.r[2]);
        u1 = _mm256_insertf128_ps(u1, M2.r[3], 1);

        let mut a0: __m256 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(0, 0, 0, 0));
        let mut a1: __m256 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(0, 0, 0, 0));
        let mut b0: __m256 = _mm256_permute2f128_ps(u0, u0, 0x00);
        let c0: __m256 = _mm256_mul_ps(a0, b0);
        let c1: __m256 = _mm256_mul_ps(a1, b0);

        a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(1, 1, 1, 1));
        a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 1, 1, 1));
        b0 = _mm256_permute2f128_ps(u0, u0, 0x11);
        let c2: __m256 = _mm256_fmadd_ps(a0, b0, c0);
        let c3: __m256 = _mm256_fmadd_ps(a1, b0, c1);

        a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(2, 2, 2, 2));
        a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 2, 2, 2));
        let mut b1: __m256 = _mm256_permute2f128_ps(u1, u1, 0x00);
        let c4: __m256 = _mm256_mul_ps(a0, b1);
        let c5: __m256 = _mm256_mul_ps(a1, b1);

        a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(3, 3, 3, 3));
        a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(3, 3, 3, 3));
        b1 = _mm256_permute2f128_ps(u1, u1, 0x11);
        let c6: __m256 = _mm256_fmadd_ps(a0, b1, c4);
        let c7: __m256 = _mm256_fmadd_ps(a1, b1, c5);

        t0 = _mm256_add_ps(c2, c6);
        t1 = _mm256_add_ps(c3, c7);

        let mut mResult: XMMATRIX = crate::undefined();
        mResult.r[0] = _mm256_castps256_ps128(t0);
        mResult.r[1] = _mm256_extractf128_ps(t0, 1);
        mResult.r[2] = _mm256_castps256_ps128(t1);
        mResult.r[3] = _mm256_extractf128_ps(t1, 1);
        return mResult;
    }

    #[cfg(all(_XM_AVX_INTRINSICS_, not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        let mut mResult: XMMATRIX = crate::undefined();

        // Splat the component X,Y,Z then W
        let mut vX: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[0])));
        let mut vY: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[1])));
        let mut vZ: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[2])));
        let mut vW: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[3])));

        // Perform the operation on the first row
        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        // Perform a binary add to reduce cumulative errors
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[0] = vX;

        // Repeat for the other 3 rows
        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[0])));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[1])));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[2])));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[3])));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[1] = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[0])));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[1])));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[2])));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[3])));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[2] = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[0])));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[1])));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[2])));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[3])));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[3] = vX;
        return mResult;
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_AVX_INTRINSICS_), not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        let mut mResult: XMMATRIX = crate::undefined();

        // Use vW to hold the original row
        let mut vW: XMVECTOR = M1.r[0];
        let mut vX: XMVECTOR = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        let mut vY: XMVECTOR = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        let mut vZ: XMVECTOR = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        // Perform the operation on the first row
        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        // Perform a binary add to reduce cumulative errors
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[0] = vX;
        // Repeat for the other 3 rows

        vW = M1.r[1];
        vX = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        vY = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        vZ = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[1] = vX;

        vW = M1.r[2];
        vX = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        vY = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        vZ = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[2] = vX;

        vW = M1.r[3];
        vX = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        vY = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        vZ = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[3] = vX;
        return mResult;
    }
}

#[test]
fn test_XMMatrixMultiply() {
    let a = XMMATRIX {
        r: [
            XMVectorSet( 1.0,  2.0,  3.0,  4.0),
            XMVectorSet( 5.0,  6.0,  7.0,  8.0),
            XMVectorSet( 9.0, 10.0, 11.0, 12.0),
            XMVectorSet(13.0, 14.0, 15.0, 16.0),
        ]
    };
    let b = XMMATRIX {
        r: [
            XMVectorSet(13.0, 14.0, 15.0, 16.0),
            XMVectorSet( 9.0, 10.0, 11.0, 12.0),
            XMVectorSet( 5.0,  6.0,  7.0,  8.0),
            XMVectorSet( 1.0,  2.0,  3.0,  4.0),
        ]
    };

    let c = XMMatrixMultiply(a, &b);

    let d = XMMATRIX {
        r: [
            XMVectorSet( 50.0,  60.0,  70.0,  80.0),
            XMVectorSet(162.0, 188.0, 214.0, 240.0),
            XMVectorSet(274.0, 316.0, 358.0, 400.0),
            XMVectorSet(386.0, 444.0, 502.0, 560.0),
        ]
    };

    unsafe {
        let mut cr = 0;
        XMVectorEqualR(&mut cr, c.r[0], d.r[0]);
        assert!(XMComparisonAllTrue(cr));

        XMVectorEqualR(&mut cr, c.r[1], d.r[1]);
        assert!(XMComparisonAllTrue(cr));

        XMVectorEqualR(&mut cr, c.r[2], d.r[2]);
        assert!(XMComparisonAllTrue(cr));

        XMVectorEqualR(&mut cr, c.r[3], d.r[3]);
        assert!(XMComparisonAllTrue(cr));
    }
}

/// Computes the transpose of the product of two matrices.
///
/// ## Parameters
///
/// `M1` First matrix to multiply.
///
/// `M2` Second matrix to multiply.
///
/// ## Return value
///
/// Returns the transpose of the product of `M1` and `M2`.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixMultiplyTranspose>
#[inline]
pub fn XMMatrixMultiplyTranspose(
    M1: FXMMATRIX,
    M2: CXMMATRIX,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut mResult: XMMATRIX = crate::undefined();
        // Cache the invariants in registers
        let mut x: f32 = M2.m[0][0];
        let mut y: f32 = M2.m[1][0];
        let mut z: f32 = M2.m[2][0];
        let mut w: f32 = M2.m[3][0];
        // Perform the operation on the first row
        mResult.m[0][0] = (M1.m[0][0] * x) + (M1.m[0][1] * y) + (M1.m[0][2] * z) + (M1.m[0][3] * w);
        mResult.m[0][1] = (M1.m[1][0] * x) + (M1.m[1][1] * y) + (M1.m[1][2] * z) + (M1.m[1][3] * w);
        mResult.m[0][2] = (M1.m[2][0] * x) + (M1.m[2][1] * y) + (M1.m[2][2] * z) + (M1.m[2][3] * w);
        mResult.m[0][3] = (M1.m[3][0] * x) + (M1.m[3][1] * y) + (M1.m[3][2] * z) + (M1.m[3][3] * w);
        // Repeat for all the other rows
        x = M2.m[0][1];
        y = M2.m[1][1];
        z = M2.m[2][1];
        w = M2.m[3][1];
        mResult.m[1][0] = (M1.m[0][0] * x) + (M1.m[0][1] * y) + (M1.m[0][2] * z) + (M1.m[0][3] * w);
        mResult.m[1][1] = (M1.m[1][0] * x) + (M1.m[1][1] * y) + (M1.m[1][2] * z) + (M1.m[1][3] * w);
        mResult.m[1][2] = (M1.m[2][0] * x) + (M1.m[2][1] * y) + (M1.m[2][2] * z) + (M1.m[2][3] * w);
        mResult.m[1][3] = (M1.m[3][0] * x) + (M1.m[3][1] * y) + (M1.m[3][2] * z) + (M1.m[3][3] * w);
        x = M2.m[0][2];
        y = M2.m[1][2];
        z = M2.m[2][2];
        w = M2.m[3][2];
        mResult.m[2][0] = (M1.m[0][0] * x) + (M1.m[0][1] * y) + (M1.m[0][2] * z) + (M1.m[0][3] * w);
        mResult.m[2][1] = (M1.m[1][0] * x) + (M1.m[1][1] * y) + (M1.m[1][2] * z) + (M1.m[1][3] * w);
        mResult.m[2][2] = (M1.m[2][0] * x) + (M1.m[2][1] * y) + (M1.m[2][2] * z) + (M1.m[2][3] * w);
        mResult.m[2][3] = (M1.m[3][0] * x) + (M1.m[3][1] * y) + (M1.m[3][2] * z) + (M1.m[3][3] * w);
        x = M2.m[0][3];
        y = M2.m[1][3];
        z = M2.m[2][3];
        w = M2.m[3][3];
        mResult.m[3][0] = (M1.m[0][0] * x) + (M1.m[0][1] * y) + (M1.m[0][2] * z) + (M1.m[0][3] * w);
        mResult.m[3][1] = (M1.m[1][0] * x) + (M1.m[1][1] * y) + (M1.m[1][2] * z) + (M1.m[1][3] * w);
        mResult.m[3][2] = (M1.m[2][0] * x) + (M1.m[2][1] * y) + (M1.m[2][2] * z) + (M1.m[2][3] * w);
        mResult.m[3][3] = (M1.m[3][0] * x) + (M1.m[3][1] * y) + (M1.m[3][2] * z) + (M1.m[3][3] * w);
        return mResult;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_AVX2_INTRINSICS_)]
    unsafe {
        let mut t0: __m256 = _mm256_castps128_ps256(M1.r[0]);
        t0 = _mm256_insertf128_ps(t0, M1.r[1], 1);
        let mut t1: __m256 = _mm256_castps128_ps256(M1.r[2]);
        t1 = _mm256_insertf128_ps(t1, M1.r[3], 1);

        let mut u0: __m256 = _mm256_castps128_ps256(M2.r[0]);
        u0 = _mm256_insertf128_ps(u0, M2.r[1], 1);
        let mut u1: __m256 = _mm256_castps128_ps256(M2.r[2]);
        u1 = _mm256_insertf128_ps(u1, M2.r[3], 1);

        let mut a0: __m256 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(0, 0, 0, 0));
        let mut a1: __m256 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(0, 0, 0, 0));
        let mut b0: __m256 = _mm256_permute2f128_ps(u0, u0, 0x00);
        let c0: __m256 = _mm256_mul_ps(a0, b0);
        let c1: __m256 = _mm256_mul_ps(a1, b0);

        a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(1, 1, 1, 1));
        a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 1, 1, 1));
        b0 = _mm256_permute2f128_ps(u0, u0, 0x11);
        let c2: __m256 = _mm256_fmadd_ps(a0, b0, c0);
        let c3: __m256 = _mm256_fmadd_ps(a1, b0, c1);

        a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(2, 2, 2, 2));
        a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 2, 2, 2));
        let mut b1: __m256 = _mm256_permute2f128_ps(u1, u1, 0x00);
        let c4: __m256 = _mm256_mul_ps(a0, b1);
        let c5: __m256 = _mm256_mul_ps(a1, b1);

        a0 = _mm256_shuffle_ps(t0, t0, _MM_SHUFFLE(3, 3, 3, 3));
        a1 = _mm256_shuffle_ps(t1, t1, _MM_SHUFFLE(3, 3, 3, 3));
        b1 = _mm256_permute2f128_ps(u1, u1, 0x11);
        let c6: __m256 = _mm256_fmadd_ps(a0, b1, c4);
        let c7: __m256 = _mm256_fmadd_ps(a1, b1, c5);

        t0 = _mm256_add_ps(c2, c6);
        t1 = _mm256_add_ps(c3, c7);

        // Transpose result
        let mut vTemp: __m256 = _mm256_unpacklo_ps(t0, t1);
        let mut vTemp2: __m256 = _mm256_unpackhi_ps(t0, t1);
        let vTemp3: __m256 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x20);
        let vTemp4: __m256 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x31);
        vTemp = _mm256_unpacklo_ps(vTemp3, vTemp4);
        vTemp2 = _mm256_unpackhi_ps(vTemp3, vTemp4);
        t0 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x20);
        t1 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x31);

        let mut mResult: XMMATRIX = crate::undefined();
        mResult.r[0] = _mm256_castps256_ps128(t0);
        mResult.r[1] = _mm256_extractf128_ps(t0, 1);
        mResult.r[2] = _mm256_castps256_ps128(t1);
        mResult.r[3] = _mm256_extractf128_ps(t1, 1);
        return mResult;
    }

    #[cfg(all(_XM_AVX_INTRINSICS_, not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        // Splat the component X,Y,Z then W
        let mut vX: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[0])));
        let mut vY: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[1])));
        let mut vZ: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[2])));
        let mut vW: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[0])[3])));

        // Perform the operation on the first row
        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        // Perform a binary add to reduce cumulative errors
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r0: XMVECTOR = vX;

        // Repeat for the other 3 rows
        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[0])));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[1])));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[2])));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[1])[3])));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r1: XMVECTOR = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[0])));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[1])));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[2])));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[2])[3])));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r2: XMVECTOR = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[0])));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[1])));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[2])));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&idx!(f32x4(M1.r[3])[3])));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r3: XMVECTOR = vX;

        // Transpose result
        // x.x,x.y,y.x,y.y
        let vTemp1: XMVECTOR = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 0, 1, 0));
        // x.z,x.w,y.z,y.w
        let vTemp3: XMVECTOR = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 2, 3, 2));
        // z.x,z.y,w.x,w.y
        let vTemp2: XMVECTOR = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(1, 0, 1, 0));
        // z.z,z.w,w.z,w.w
        let vTemp4: XMVECTOR = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 2, 3, 2));

        let mut mResult: XMMATRIX = crate::undefined();
        // x.x,y.x,z.x,w.x
        mResult.r[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
        // x.y,y.y,z.y,w.y
        mResult.r[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
        // x.z,y.z,z.z,w.z
        mResult.r[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
        // x.w,y.w,z.w,w.w
        mResult.r[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));
        return mResult;
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_AVX_INTRINSICS_), not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        // Use vW to hold the original row
        let mut vW: XMVECTOR = M1.r[0];
        let mut vX: XMVECTOR = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        let mut vY: XMVECTOR = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        let mut vZ: XMVECTOR = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        // Perform the operation on the first row
        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        // Perform a binary add to reduce cumulative errors
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r0: XMVECTOR = vX;
        // Repeat for the other 3 rows

        vW = M1.r[1];
        vX = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        vY = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        vZ = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r1: XMVECTOR = vX;

        vW = M1.r[2];
        vX = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        vY = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        vZ = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r2: XMVECTOR = vX;

        vW = M1.r[3];
        vX = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(0, 0, 0, 0));
        vY = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(1, 1, 1, 1));
        vZ = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(2, 2, 2, 2));
        vW = XM_PERMUTE_PS!(vW, _MM_SHUFFLE(3, 3, 3, 3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);
        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r3: XMVECTOR = vX;

        // Transpose result
        // x.x,x.y,y.x,y.y
        let vTemp1: XMVECTOR = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 0, 1, 0));
        // x.z,x.w,y.z,y.w
        let vTemp3: XMVECTOR = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 2, 3, 2));
        // z.x,z.y,w.x,w.y
        let vTemp2: XMVECTOR = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(1, 0, 1, 0));
        // z.z,z.w,w.z,w.w
        let vTemp4: XMVECTOR = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 2, 3, 2));

        let mut mResult: XMMATRIX = crate::undefined();
        // x.x,y.x,z.x,w.x
        mResult.r[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
        // x.y,y.y,z.y,w.y
        mResult.r[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
        // x.z,y.z,z.z,w.z
        mResult.r[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
        // x.w,y.w,z.w,w.w
        mResult.r[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));
        return mResult;
    }
}

#[test]
fn test_XMMatrixMultiplyTranspose() {
    let a = XMMATRIX {
        r: [
            XMVectorSet( 1.0,  2.0,  3.0,  4.0),
            XMVectorSet( 5.0,  6.0,  7.0,  8.0),
            XMVectorSet( 9.0, 10.0, 11.0, 12.0),
            XMVectorSet(13.0, 14.0, 15.0, 16.0),
        ]
    };
    let b = XMMATRIX {
        r: [
            XMVectorSet(13.0, 14.0, 15.0, 16.0),
            XMVectorSet( 9.0, 10.0, 11.0, 12.0),
            XMVectorSet( 5.0,  6.0,  7.0,  8.0),
            XMVectorSet( 1.0,  2.0,  3.0,  4.0),
        ]
    };

    let c = XMMatrixMultiplyTranspose(a, &b);

    let d = XMMATRIX {
        r: [
            XMVectorSet( 50.0,  60.0,  70.0,  80.0),
            XMVectorSet(162.0, 188.0, 214.0, 240.0),
            XMVectorSet(274.0, 316.0, 358.0, 400.0),
            XMVectorSet(386.0, 444.0, 502.0, 560.0),
        ]
    };

    let d = XMMatrixTranspose(d);

    unsafe {
        let mut cr = 0;
        XMVectorEqualR(&mut cr, c.r[0], d.r[0]);
        assert!(XMComparisonAllTrue(cr));

        XMVectorEqualR(&mut cr, c.r[1], d.r[1]);
        assert!(XMComparisonAllTrue(cr));

        XMVectorEqualR(&mut cr, c.r[2], d.r[2]);
        assert!(XMComparisonAllTrue(cr));

        XMVectorEqualR(&mut cr, c.r[3], d.r[3]);
        assert!(XMComparisonAllTrue(cr));
    }
}

/// Computes the transpose of a matrix.
///
/// ## Parameters
///
/// `M` Matrix to transpose.
///
/// ## Return value
///
/// Returns the transpose of M.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixTranspose>
#[inline]
pub fn XMMatrixTranspose(
    M: FXMMATRIX,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        // Original matrix:
        //
        //     m00m01m02m03
        //     m10m11m12m13
        //     m20m21m22m23
        //     m30m31m32m33

        let mut P: XMMATRIX = crate::undefined();
        P.r[0] = XMVectorMergeXY(M.r[0], M.r[2]); // m00m20m01m21
        P.r[1] = XMVectorMergeXY(M.r[1], M.r[3]); // m10m30m11m31
        P.r[2] = XMVectorMergeZW(M.r[0], M.r[2]); // m02m22m03m23
        P.r[3] = XMVectorMergeZW(M.r[1], M.r[3]); // m12m32m13m33

        let mut MT: XMMATRIX = crate::undefined();
        MT.r[0] = XMVectorMergeXY(P.r[0], P.r[1]); // m00m10m20m30
        MT.r[1] = XMVectorMergeZW(P.r[0], P.r[1]); // m01m11m21m31
        MT.r[2] = XMVectorMergeXY(P.r[2], P.r[3]); // m02m12m22m32
        MT.r[3] = XMVectorMergeZW(P.r[2], P.r[3]); // m03m13m23m33
        return MT;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_AVX2_INTRINSICS_)]
    unsafe {
        let mut t0: __m256 = _mm256_castps128_ps256(M.r[0]);
        t0 = _mm256_insertf128_ps(t0, M.r[1], 1);
        let mut t1: __m256 = _mm256_castps128_ps256(M.r[2]);
        t1 = _mm256_insertf128_ps(t1, M.r[3], 1);

        let mut vTemp: __m256 = _mm256_unpacklo_ps(t0, t1);
        let mut vTemp2: __m256 = _mm256_unpackhi_ps(t0, t1);
        let vTemp3: __m256 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x20);
        let vTemp4: __m256 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x31);
        vTemp = _mm256_unpacklo_ps(vTemp3, vTemp4);
        vTemp2 = _mm256_unpackhi_ps(vTemp3, vTemp4);
        t0 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x20);
        t1 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x31);

        let mut mResult: XMMATRIX = crate::undefined();
        mResult.r[0] = _mm256_castps256_ps128(t0);
        mResult.r[1] = _mm256_extractf128_ps(t0, 1);
        mResult.r[2] = _mm256_castps256_ps128(t1);
        mResult.r[3] = _mm256_extractf128_ps(t1, 1);
        return mResult;
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        // x.x,x.y,y.x,y.y
        let vTemp1: XMVECTOR = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(1, 0, 1, 0));
        // x.z,x.w,y.z,y.w
        let vTemp3: XMVECTOR = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(3, 2, 3, 2));
        // z.x,z.y,w.x,w.y
        let vTemp2: XMVECTOR = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(1, 0, 1, 0));
        // z.z,z.w,w.z,w.w
        let vTemp4: XMVECTOR = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(3, 2, 3, 2));

        let mut mResult: XMMATRIX = crate::undefined();
        // x.x,y.x,z.x,w.x
        mResult.r[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
        // x.y,y.y,z.y,w.y
        mResult.r[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
        // x.z,y.z,z.z,w.z
        mResult.r[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
        // x.w,y.w,z.w,w.w
        mResult.r[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));
        return mResult;
    }
}

/// Computes the inverse of a matrix.
///
/// ## Parameters
///
/// `pDeterminant` Address of a vector, each of whose components receives the determinant of `M`. This parameter may be `None`
///  if the determinant is not desired.
///
/// `M` Matrix to invert.
///
/// ## Return value
///
/// Returns the matrix inverse of `M`. If there is no inverse (that is, if the determinant is `0`), XMMatrixInverse
/// returns an infinite matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixInverse>
#[inline]
pub fn XMMatrixInverse(
    pDeterminant: Option<&mut XMVECTOR>,
    M: FXMMATRIX,
) -> XMMATRIX
{
    #[cfg(any(_XM_NO_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        let MT: XMMATRIX = XMMatrixTranspose(M);

        let mut V0: [XMVECTOR; 4] = crate::undefined();
        let mut V1: [XMVECTOR; 4] = crate::undefined();
        V0[0] = <(XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[2]);
        V1[0] = <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_W)>::XMVectorSwizzle(MT.r[3]);
        V0[1] = <(XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[0]);
        V1[1] = <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_W)>::XMVectorSwizzle(MT.r[1]);
        V1[2] = <(XM_PERMUTE_0Y, XM_PERMUTE_0W, XM_PERMUTE_1Y, XM_PERMUTE_1W)>::XMVectorPermute(MT.r[3], MT.r[1]);
        V0[2] = <(XM_PERMUTE_0X, XM_PERMUTE_0Z, XM_PERMUTE_1X, XM_PERMUTE_1Z)>::XMVectorPermute(MT.r[2], MT.r[0]);

        let mut D0: XMVECTOR = XMVectorMultiply(V0[0], V1[0]);
        let mut D1: XMVECTOR = XMVectorMultiply(V0[1], V1[1]);
        let mut D2: XMVECTOR = XMVectorMultiply(V0[2], V1[2]);

        V0[0] = <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_W)>::XMVectorSwizzle(MT.r[2]);
        V1[0] = <(XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[3]);
        V0[1] = <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_W)>::XMVectorSwizzle(MT.r[0]);
        V1[1] = <(XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[1]);
        V0[2] = <(XM_PERMUTE_0Y, XM_PERMUTE_0W, XM_PERMUTE_1Y, XM_PERMUTE_1W)>::XMVectorPermute(MT.r[2], MT.r[0]);
        V1[2] = <(XM_PERMUTE_0X, XM_PERMUTE_0Z, XM_PERMUTE_1X, XM_PERMUTE_1Z)>::XMVectorPermute(MT.r[3], MT.r[1]);

        D0 = XMVectorNegativeMultiplySubtract(V0[0], V1[0], D0);
        D1 = XMVectorNegativeMultiplySubtract(V0[1], V1[1], D1);
        D2 = XMVectorNegativeMultiplySubtract(V0[2], V1[2], D2);

        V0[0] = <(XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[1]);
        V1[0] = <(XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_0W, XM_PERMUTE_0X)>::XMVectorPermute(D0, D2);
        V0[1] = <(XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(MT.r[0]);
        V1[1] = <(XM_PERMUTE_0W, XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_0Z)>::XMVectorPermute(D0, D2);
        V0[2] = <(XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[3]);
        V1[2] = <(XM_PERMUTE_1W, XM_PERMUTE_0Y, XM_PERMUTE_0W, XM_PERMUTE_0X)>::XMVectorPermute(D1, D2);
        V0[3] = <(XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(MT.r[2]);
        V1[3] = <(XM_PERMUTE_0W, XM_PERMUTE_1W, XM_PERMUTE_0Y, XM_PERMUTE_0Z)>::XMVectorPermute(D1, D2);

        let mut C0: XMVECTOR = XMVectorMultiply(V0[0], V1[0]);
        let mut C2: XMVECTOR = XMVectorMultiply(V0[1], V1[1]);
        let mut C4: XMVECTOR = XMVectorMultiply(V0[2], V1[2]);
        let mut C6: XMVECTOR = XMVectorMultiply(V0[3], V1[3]);

        V0[0] = <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Y, XM_SWIZZLE_Z)>::XMVectorSwizzle(MT.r[1]);
        V1[0] = <(XM_PERMUTE_0W, XM_PERMUTE_0X, XM_PERMUTE_0Y, XM_PERMUTE_1X)>::XMVectorPermute(D0, D2);
        V0[1] = <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[0]);
        V1[1] = <(XM_PERMUTE_0Z, XM_PERMUTE_0Y, XM_PERMUTE_1X, XM_PERMUTE_0X)>::XMVectorPermute(D0, D2);
        V0[2] = <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Y, XM_SWIZZLE_Z)>::XMVectorSwizzle(MT.r[3]);
        V1[2] = <(XM_PERMUTE_0W, XM_PERMUTE_0X, XM_PERMUTE_0Y, XM_PERMUTE_1Z)>::XMVectorPermute(D1, D2);
        V0[3] = <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_Y)>::XMVectorSwizzle(MT.r[2]);
        V1[3] = <(XM_PERMUTE_0Z, XM_PERMUTE_0Y, XM_PERMUTE_1Z, XM_PERMUTE_0X)>::XMVectorPermute(D1, D2);

        C0 = XMVectorNegativeMultiplySubtract(V0[0], V1[0], C0);
        C2 = XMVectorNegativeMultiplySubtract(V0[1], V1[1], C2);
        C4 = XMVectorNegativeMultiplySubtract(V0[2], V1[2], C4);
        C6 = XMVectorNegativeMultiplySubtract(V0[3], V1[3], C6);

        V0[0] = <(XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_X)>::XMVectorSwizzle(MT.r[1]);
        V1[0] = <(XM_PERMUTE_0Z, XM_PERMUTE_1Y, XM_PERMUTE_1X, XM_PERMUTE_0Z)>::XMVectorPermute(D0, D2);
        V0[1] = <(XM_SWIZZLE_Y, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Z)>::XMVectorSwizzle(MT.r[0]);
        V1[1] = <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_1X)>::XMVectorPermute(D0, D2);
        V0[2] = <(XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_X)>::XMVectorSwizzle(MT.r[3]);
        V1[2] = <(XM_PERMUTE_0Z, XM_PERMUTE_1W, XM_PERMUTE_1Z, XM_PERMUTE_0Z)>::XMVectorPermute(D1, D2);
        V0[3] = <(XM_SWIZZLE_Y, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Z)>::XMVectorSwizzle(MT.r[2]);
        V1[3] = <(XM_PERMUTE_1W, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_1Z)>::XMVectorPermute(D1, D2);

        let C1: XMVECTOR = XMVectorNegativeMultiplySubtract(V0[0], V1[0], C0);
        C0 = XMVectorMultiplyAdd(V0[0], V1[0], C0);
        let C3: XMVECTOR = XMVectorMultiplyAdd(V0[1], V1[1], C2);
        C2 = XMVectorNegativeMultiplySubtract(V0[1], V1[1], C2);
        let C5: XMVECTOR = XMVectorNegativeMultiplySubtract(V0[2], V1[2], C4);
        C4 = XMVectorMultiplyAdd(V0[2], V1[2], C4);
        let C7: XMVECTOR = XMVectorMultiplyAdd(V0[3], V1[3], C6);
        C6 = XMVectorNegativeMultiplySubtract(V0[3], V1[3], C6);

        let mut R: XMMATRIX = crate::undefined();
        R.r[0] = XMVectorSelect(C0, C1, g_XMSelect0101.v);
        R.r[1] = XMVectorSelect(C2, C3, g_XMSelect0101.v);
        R.r[2] = XMVectorSelect(C4, C5, g_XMSelect0101.v);
        R.r[3] = XMVectorSelect(C6, C7, g_XMSelect0101.v);

        let Determinant: XMVECTOR = XMVector4Dot(R.r[0], MT.r[0]);

        match pDeterminant {
            Some(determinant) => *determinant = Determinant,
            None => ()
        };

        let Reciprocal: XMVECTOR = XMVectorReciprocal(Determinant);

        let mut Result: XMMATRIX = crate::undefined();
        Result.r[0] = XMVectorMultiply(R.r[0], Reciprocal);
        Result.r[1] = XMVectorMultiply(R.r[1], Reciprocal);
        Result.r[2] = XMVectorMultiply(R.r[2], Reciprocal);
        Result.r[3] = XMVectorMultiply(R.r[3], Reciprocal);
        return Result;
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // Transpose matrix
        let vTemp1: XMVECTOR = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(1, 0, 1, 0));
        let vTemp3: XMVECTOR = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(3, 2, 3, 2));
        let vTemp2: XMVECTOR = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(1, 0, 1, 0));
        let vTemp4: XMVECTOR = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(3, 2, 3, 2));

        let mut MT: XMMATRIX = crate::undefined();
        MT.r[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
        MT.r[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
        MT.r[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
        MT.r[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));

        let mut V00: XMVECTOR = XM_PERMUTE_PS!(MT.r[2], _MM_SHUFFLE(1, 1, 0, 0));
        let mut V10: XMVECTOR = XM_PERMUTE_PS!(MT.r[3], _MM_SHUFFLE(3, 2, 3, 2));
        let mut V01: XMVECTOR = XM_PERMUTE_PS!(MT.r[0], _MM_SHUFFLE(1, 1, 0, 0));
        let mut V11: XMVECTOR = XM_PERMUTE_PS!(MT.r[1], _MM_SHUFFLE(3, 2, 3, 2));
        let mut V02: XMVECTOR = _mm_shuffle_ps(MT.r[2], MT.r[0], _MM_SHUFFLE(2, 0, 2, 0));
        let mut V12: XMVECTOR = _mm_shuffle_ps(MT.r[3], MT.r[1], _MM_SHUFFLE(3, 1, 3, 1));

        let mut D0: XMVECTOR = _mm_mul_ps(V00, V10);
        let mut D1: XMVECTOR = _mm_mul_ps(V01, V11);
        let mut D2: XMVECTOR = _mm_mul_ps(V02, V12);

        V00 = XM_PERMUTE_PS!(MT.r[2], _MM_SHUFFLE(3, 2, 3, 2));
        V10 = XM_PERMUTE_PS!(MT.r[3], _MM_SHUFFLE(1, 1, 0, 0));
        V01 = XM_PERMUTE_PS!(MT.r[0], _MM_SHUFFLE(3, 2, 3, 2));
        V11 = XM_PERMUTE_PS!(MT.r[1], _MM_SHUFFLE(1, 1, 0, 0));
        V02 = _mm_shuffle_ps(MT.r[2], MT.r[0], _MM_SHUFFLE(3, 1, 3, 1));
        V12 = _mm_shuffle_ps(MT.r[3], MT.r[1], _MM_SHUFFLE(2, 0, 2, 0));

        D0 = XM_FNMADD_PS!(V00, V10, D0);
        D1 = XM_FNMADD_PS!(V01, V11, D1);
        D2 = XM_FNMADD_PS!(V02, V12, D2);
        // V11 = D0Y,D0W,D2Y,D2Y
        V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 1, 3, 1));
        V00 = XM_PERMUTE_PS!(MT.r[1], _MM_SHUFFLE(1, 0, 2, 1));
        V10 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(0, 3, 0, 2));
        V01 = XM_PERMUTE_PS!(MT.r[0], _MM_SHUFFLE(0, 1, 0, 2));
        V11 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(2, 1, 2, 1));
        // V13 = D1Y,D1W,D2W,D2W
        let mut V13: XMVECTOR = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 3, 3, 1));
        V02 = XM_PERMUTE_PS!(MT.r[3], _MM_SHUFFLE(1, 0, 2, 1));
        V12 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(0, 3, 0, 2));
        let mut V03: XMVECTOR = XM_PERMUTE_PS!(MT.r[2], _MM_SHUFFLE(0, 1, 0, 2));
        V13 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(2, 1, 2, 1));

        let mut C0: XMVECTOR = _mm_mul_ps(V00, V10);
        let mut C2: XMVECTOR = _mm_mul_ps(V01, V11);
        let mut C4: XMVECTOR = _mm_mul_ps(V02, V12);
        let mut C6: XMVECTOR = _mm_mul_ps(V03, V13);

        // V11 = D0X,D0Y,D2X,D2X
        V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(0, 0, 1, 0));
        V00 = XM_PERMUTE_PS!(MT.r[1], _MM_SHUFFLE(2, 1, 3, 2));
        V10 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(2, 1, 0, 3));
        V01 = XM_PERMUTE_PS!(MT.r[0], _MM_SHUFFLE(1, 3, 2, 3));
        V11 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(0, 2, 1, 2));
        // V13 = D1X,D1Y,D2Z,D2Z
        V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(2, 2, 1, 0));
        V02 = XM_PERMUTE_PS!(MT.r[3], _MM_SHUFFLE(2, 1, 3, 2));
        V12 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(2, 1, 0, 3));
        V03 = XM_PERMUTE_PS!(MT.r[2], _MM_SHUFFLE(1, 3, 2, 3));
        V13 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(0, 2, 1, 2));

        C0 = XM_FNMADD_PS!(V00, V10, C0);
        C2 = XM_FNMADD_PS!(V01, V11, C2);
        C4 = XM_FNMADD_PS!(V02, V12, C4);
        C6 = XM_FNMADD_PS!(V03, V13, C6);

        V00 = XM_PERMUTE_PS!(MT.r[1], _MM_SHUFFLE(0, 3, 0, 3));
        // V10 = D0Z,D0Z,D2X,D2Y
        V10 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 2, 2));
        V10 = XM_PERMUTE_PS!(V10, _MM_SHUFFLE(0, 2, 3, 0));
        V01 = XM_PERMUTE_PS!(MT.r[0], _MM_SHUFFLE(2, 0, 3, 1));
        // V11 = D0X,D0W,D2X,D2Y
        V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 3, 0));
        V11 = XM_PERMUTE_PS!(V11, _MM_SHUFFLE(2, 1, 0, 3));
        V02 = XM_PERMUTE_PS!(MT.r[3], _MM_SHUFFLE(0, 3, 0, 3));
        // V12 = D1Z,D1Z,D2Z,D2W
        V12 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 2, 2));
        V12 = XM_PERMUTE_PS!(V12, _MM_SHUFFLE(0, 2, 3, 0));
        V03 = XM_PERMUTE_PS!(MT.r[2], _MM_SHUFFLE(2, 0, 3, 1));
        // V13 = D1X,D1W,D2Z,D2W
        V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 3, 0));
        V13 = XM_PERMUTE_PS!(V13, _MM_SHUFFLE(2, 1, 0, 3));

        V00 = _mm_mul_ps(V00, V10);
        V01 = _mm_mul_ps(V01, V11);
        V02 = _mm_mul_ps(V02, V12);
        V03 = _mm_mul_ps(V03, V13);
        let C1: XMVECTOR = _mm_sub_ps(C0, V00);
        C0 = _mm_add_ps(C0, V00);
        let C3: XMVECTOR = _mm_add_ps(C2, V01);
        C2 = _mm_sub_ps(C2, V01);
        let C5: XMVECTOR = _mm_sub_ps(C4, V02);
        C4 = _mm_add_ps(C4, V02);
        let C7: XMVECTOR = _mm_add_ps(C6, V03);
        C6 = _mm_sub_ps(C6, V03);

        C0 = _mm_shuffle_ps(C0, C1, _MM_SHUFFLE(3, 1, 2, 0));
        C2 = _mm_shuffle_ps(C2, C3, _MM_SHUFFLE(3, 1, 2, 0));
        C4 = _mm_shuffle_ps(C4, C5, _MM_SHUFFLE(3, 1, 2, 0));
        C6 = _mm_shuffle_ps(C6, C7, _MM_SHUFFLE(3, 1, 2, 0));
        C0 = XM_PERMUTE_PS!(C0, _MM_SHUFFLE(3, 1, 2, 0));
        C2 = XM_PERMUTE_PS!(C2, _MM_SHUFFLE(3, 1, 2, 0));
        C4 = XM_PERMUTE_PS!(C4, _MM_SHUFFLE(3, 1, 2, 0));
        C6 = XM_PERMUTE_PS!(C6, _MM_SHUFFLE(3, 1, 2, 0));
        // Get the determinant
        let mut vTemp: XMVECTOR = XMVector4Dot(C0, MT.r[0]);

        if let Some(determinant) = pDeterminant {
            *determinant = vTemp;
        }

        vTemp = _mm_div_ps(g_XMOne.v, vTemp);
        let mut mResult: XMMATRIX = crate::undefined();
        mResult.r[0] = _mm_mul_ps(C0, vTemp);
        mResult.r[1] = _mm_mul_ps(C2, vTemp);
        mResult.r[2] = _mm_mul_ps(C4, vTemp);
        mResult.r[3] = _mm_mul_ps(C6, vTemp);
        return mResult;
    }
}

/// XMMatrixVectorTensorProduct
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixVectorTensorProduct>
#[inline]
pub fn XMMatrixVectorTensorProduct(
    V1: FXMVECTOR,
    V2: FXMVECTOR
) -> XMMATRIX
{
    unsafe {
        let mut mResult: XMMATRIX = crate::undefined();
        type _0 = XM_SWIZZLE_X;
        type _1 = XM_SWIZZLE_Y;
        type _2 = XM_SWIZZLE_Z;
        type _3 = XM_SWIZZLE_W;
        mResult.r[0] = XMVectorMultiply(<(_0, _0, _0, _0)>::XMVectorSwizzle(V1), V2);
        mResult.r[1] = XMVectorMultiply(<(_1, _1, _1, _1)>::XMVectorSwizzle(V1), V2);
        mResult.r[2] = XMVectorMultiply(<(_2, _2, _2, _2)>::XMVectorSwizzle(V1), V2);
        mResult.r[3] = XMVectorMultiply(<(_3, _3, _3, _3)>::XMVectorSwizzle(V1), V2);
        return mResult;
    }
}

/// Computes the determinant of a matrix.
///
/// ## Parameters
///
/// `M` Matrix from which to compute the determinant.
///
/// ## Return value
///
/// Returns a vector. The determinant of `M` is replicated into each component.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixDeterminant>
#[inline]
pub fn XMMatrixDeterminant(
    M: FXMMATRIX,
) -> FXMVECTOR
{
    unsafe {
        const Sign: XMVECTORF32 = XMVECTORF32 { f: [ 1.0, -1.0, 1.0, -1.0 ] };

        let mut V0: XMVECTOR = <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_X)>::XMVectorSwizzle(M.r[2]);
        let mut V1: XMVECTOR = <(XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(M.r[3]);
        let mut V2: XMVECTOR = <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_X)>::XMVectorSwizzle(M.r[2]);
        let mut V3: XMVECTOR = <(XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(M.r[3]);
        let mut V4: XMVECTOR = <(XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(M.r[2]);
        let mut V5: XMVECTOR = <(XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(M.r[3]);

        let mut P0: XMVECTOR = XMVectorMultiply(V0, V1);
        let mut P1: XMVECTOR = XMVectorMultiply(V2, V3);
        let mut P2: XMVECTOR = XMVectorMultiply(V4, V5);

        V0 = <(XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(M.r[2]);
        V1 = <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_X)>::XMVectorSwizzle(M.r[3]);
        V2 = <(XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(M.r[2]);
        V3 = <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_X)>::XMVectorSwizzle(M.r[3]);
        V4 = <(XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(M.r[2]);
        V5 = <(XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(M.r[3]);

        P0 = XMVectorNegativeMultiplySubtract(V0, V1, P0);
        P1 = XMVectorNegativeMultiplySubtract(V2, V3, P1);
        P2 = XMVectorNegativeMultiplySubtract(V4, V5, P2);

        V0 = <(XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(M.r[1]);
        V1 = <(XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_Y)>::XMVectorSwizzle(M.r[1]);
        V2 = <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_X)>::XMVectorSwizzle(M.r[1]);

        let S: XMVECTOR = XMVectorMultiply(M.r[0], Sign.v);
        let mut R: XMVECTOR = XMVectorMultiply(V0, P0);
        R = XMVectorNegativeMultiplySubtract(V1, P1, R);
        R = XMVectorMultiplyAdd(V2, P2, R);

        return XMVector4Dot(S, R);
    }
}

/// Breaks down a general 3D transformation matrix into its scalar, rotational, and translational components.
///
/// ## Parameters
///
/// `outScale` Pointer to the output XMVECTOR that contains scaling factors applied along the `x`, `y`, and `z-axes`.
///
/// `outRotQuat` Pointer to the XMVECTOR quaternion that describes the rotation.
///
/// `outTrans` Pointer to the XMVECTOR vector that describes a translation along the `x`, `y`, and `z-axes`.
///
/// `M` Pointer to an input XMMATRIX matrix to decompose.
///
/// ## Return value
///
/// If the function succeeds, the return value is `true`. If the function fails, the return value is `false`.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixDecompose>
#[inline]
pub fn XMMatrixDecompose(
    outScale: &mut XMVECTOR,
    outRotQuat: &mut XMVECTOR,
    outTrans: &mut XMVECTOR,
    M: FXMMATRIX,
) -> bool
{
    macro_rules! XM3RANKDECOMPOSE {
        ($a:expr, $b:expr, $c:expr, $x:expr, $y:expr, $z:expr) => {
            if $x < $y {
                if $y < $z {
                    $a = 2;
                    $b = 1;
                    $c = 0;
                } else {
                    $a = 1;

                    if $x < $z {
                        $b = 2;
                        $c = 0;
                    } else {
                        $b = 0;
                        $c = 2;
                    }
                }
            } else {
                if $x < $z {
                    $a = 2;
                    $b = 0;
                    $c = 1;
                } else {
                    $a = 0;

                    if $y < $z {
                        $b = 2;
                        $c = 1;
                    } else {
                        $b = 1;
                        $c = 2;
                    }
                }
            }
        }
    }

    const XM3_DECOMP_EPSILON: f32 = 0.0001;

    unsafe {
        // PERFORMANCE: We're using a static here insteads of const due to the references
        static pvCanonicalBasis: [&XMVECTOR; 3] = unsafe {[
            &g_XMIdentityR0.v,
            &g_XMIdentityR2.v,
            &g_XMIdentityR3.v,
        ]};

        // Get the translation
        *outTrans = M.r[3];

        let mut ppvBasis: [*mut XMVECTOR; 3] = [std::ptr::null_mut(); 3];
        let mut matTemp: XMMATRIX = crate::undefined();
        ppvBasis[0] = &mut matTemp.r[0];
        ppvBasis[1] = &mut matTemp.r[1];
        ppvBasis[2] = &mut matTemp.r[2];

        matTemp.r[0] = M.r[0];
        matTemp.r[1] = M.r[1];
        matTemp.r[2] = M.r[2];
        matTemp.r[3] = g_XMIdentityR3.v;

        let pfScales = mem::transmute::<_, *mut f32>(outScale);
        XMVectorGetXPtr(&mut *pfScales.add(0), XMVector3Length(*ppvBasis[0]));
        XMVectorGetXPtr(&mut *pfScales.add(1), XMVector3Length(*ppvBasis[1]));
        XMVectorGetXPtr(&mut *pfScales.add(2), XMVector3Length(*ppvBasis[2]));
        *pfScales.add(3) = 0.0;

        let a: usize;
        let b: usize;
        let c: usize;

        let x = *pfScales.add(0);
        let y = *pfScales.add(1);
        let z = *pfScales.add(2);
        XM3RANKDECOMPOSE!(a, b, c, x, y, z);

        if (*pfScales.add(a) < XM3_DECOMP_EPSILON)
        {
            *ppvBasis[a] = *pvCanonicalBasis[a];
        }

        *ppvBasis[a] = XMVector3Normalize(*ppvBasis[a]);

        if (*pfScales.add(b) < XM3_DECOMP_EPSILON)
        {
            let _aa: usize;
            let _bb: usize;
            let cc: usize;

            let fAbsX = fabsf(XMVectorGetX(*ppvBasis[a]));
            let fAbsY = fabsf(XMVectorGetY(*ppvBasis[a]));
            let fAbsZ = fabsf(XMVectorGetZ(*ppvBasis[a]));

            XM3RANKDECOMPOSE!(_aa, _bb, cc, fAbsX, fAbsY, fAbsZ);

            *ppvBasis[b] = XMVector3Cross(*ppvBasis[a], *pvCanonicalBasis[cc]);
        }

        *ppvBasis[b] = XMVector3Normalize(*ppvBasis[b]);

        if (*pfScales.add(c) < XM3_DECOMP_EPSILON)
        {
            *ppvBasis[c] = XMVector3Cross(*ppvBasis[a], *ppvBasis[b]);
        }

        *ppvBasis[c] = XMVector3Normalize(*ppvBasis[c]);

        let mut fDet: f32 = XMVectorGetX(XMMatrixDeterminant(matTemp));

        // use Kramer's rule to check for handedness of coordinate system
        if (fDet < 0.0)
        {
            // switch coordinate system by negating the scale and inverting the basis vector on the x-axis
            *pfScales.add(a) = -(*pfScales.add(a));
            *ppvBasis[a] = XMVectorNegate(*ppvBasis[a]);

            fDet = -fDet;
        }

        fDet -= 1.0;
        fDet *= fDet;

        if (XM3_DECOMP_EPSILON < fDet)
        {
            // Non-SRT matrix encountered
            return false;
        }

        // generate the quaternion from the matrix
        *outRotQuat = XMQuaternionRotationMatrix(matTemp);
        return true;
    }
}

#[test]
fn test_XMMatrixDecompose() {
    let scaling_origin = XMVectorSet(0.0, 0.0, 0.0, 0.0);
    let scaling_orientation_quaternion = XMQuaternionRotationRollPitchYaw(0.1 ,0.2, 0.3);
    let scaling = XMVectorSet(1.1, 1.2, 1.3, 0.0);
    let rotation_origin = XMVectorSet(0.0, 0.0, 0.0, 0.0);
    let rotation_quaternion = XMQuaternionRotationRollPitchYaw(0.4 ,0.5, 0.6);
    let translation = XMVectorSet(7.0, 8.0, 9.0, 0.0);

    let transform = XMMatrixTransformation(
        scaling_origin,
        scaling_orientation_quaternion,
        scaling,
        rotation_origin,
        rotation_quaternion,
        translation
    );

    let mut out_scale = XMVectorZero();
    let mut out_rot_quat = XMVectorZero();
    let mut out_trans = XMVectorZero();

    assert!(XMMatrixDecompose(&mut out_scale, &mut out_rot_quat, &mut out_trans, transform));

    let epsilon = XMVectorReplicate(1.0e-1);

    assert!(XMVector3NearEqual(out_scale, scaling, epsilon));
    assert!(XMVector4NearEqual(out_rot_quat, rotation_quaternion, epsilon));
    assert!(XMVector3NearEqual(out_trans, translation, epsilon));
}

/// Builds the identity matrix.
///
/// ## Parameters
///
/// This function has no parameters.
///
/// ## Return value
///
/// Returns the identity matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixIdentity>
#[inline]
pub fn XMMatrixIdentity() -> XMMATRIX
{
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = g_XMIdentityR0.v;
        M.r[1] = g_XMIdentityR1.v;
        M.r[2] = g_XMIdentityR2.v;
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Creates a matrix with float values.
///
/// ## Parameters
///
/// `m00` Value to assign to the `(0,0)` element.
///
/// `m01` Value to assign to the `(0,1)` element.
///
/// `m02` Value to assign to the `(0,2)` element.
///
/// `m03` Value to assign to the `(0,3)` element.
///
/// `m10` Value to assign to the `(1,0)` element.
///
/// `m11` Value to assign to the `(1,1)` element.
///
/// `m12` Value to assign to the `(1,2)` element.
///
/// `m13` Value to assign to the `(1,3)` element.
///
/// `m20` Value to assign to the `(2,0)` element.
///
/// `m21` Value to assign to the `(2,1)` element.
///
/// `m22` Value to assign to the `(2,2)` element.
///
/// `m23` Value to assign to the `(2,3)` element.
///
/// `m30` Value to assign to the `(3,0)` element.
///
/// `m31` Value to assign to the `(3,1)` element.
///
/// `m32` Value to assign to the `(3,2)` element.
///
/// `m33` Value to assign to the `(3,3)` element.
///
/// ## Return value
///
/// Returns the XMMATRIX with the specified elements.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixSet>
#[inline]
pub fn XMMatrixSet(
    m00: f32, m01: f32, m02: f32, m03: f32,
    m10: f32, m11: f32, m12: f32, m13: f32,
    m20: f32, m21: f32, m22: f32, m23: f32,
    m30: f32, m31: f32, m32: f32, m33: f32
) -> XMMATRIX
{
    let mut M: XMMATRIX = unsafe { crate::undefined() };

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        M.m[0][0] = m00; M.m[0][1] = m01; M.m[0][2] = m02; M.m[0][3] = m03;
        M.m[1][0] = m10; M.m[1][1] = m11; M.m[1][2] = m12; M.m[1][3] = m13;
        M.m[2][0] = m20; M.m[2][1] = m21; M.m[2][2] = m22; M.m[2][3] = m23;
        M.m[3][0] = m30; M.m[3][1] = m31; M.m[3][2] = m32; M.m[3][3] = m33;
    }

    #[cfg(not(_XM_NO_INTRINSICS_))]
    unsafe {
        M.r[0] = XMVectorSet(m00, m01, m02, m03);
        M.r[1] = XMVectorSet(m10, m11, m12, m13);
        M.r[2] = XMVectorSet(m20, m21, m22, m23);
        M.r[3] = XMVectorSet(m30, m31, m32, m33);
    }

    return M;
}

/// Builds a translation matrix from the specified offsets.
///
/// ## Parameters
///
/// `OffsetX` Translation along the `x-axis`.
///
/// `OffsetY` Translation along the `y-axis`.
///
/// `OffsetZ` Translation along the `z-axis`.
///
/// ## Return value
///
/// Returns the translation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixTranslation>
#[inline]
pub fn XMMatrixTranslation(
    OffsetX: f32,
    OffsetY: f32,
    OffsetZ: f32,
) -> XMMATRIX
{
    let mut M: XMMATRIX = unsafe { crate::undefined() };

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        M.m[0][0] = 1.0;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = 1.0;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = 1.0;
        M.m[2][3] = 0.0;

        M.m[3][0] = OffsetX;
        M.m[3][1] = OffsetY;
        M.m[3][2] = OffsetZ;
        M.m[3][3] = 1.0;
    }

    #[cfg(any(_XM_SSE_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        M.r[0] = g_XMIdentityR0.v;
        M.r[1] = g_XMIdentityR1.v;
        M.r[2] = g_XMIdentityR2.v;
        M.r[3] = XMVectorSet(OffsetX, OffsetY, OffsetZ, 1.0);
    }

    return M;
}

/// Builds a translation matrix from a vector.
///
/// ## Parameters
///
/// `Offset` 3D vector describing the translations along the `x-axis`, `y-axis`, and `z-axis`.
///
/// ## Return value
///
/// Returns the translation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixTranslationFromVector>
#[inline]
pub fn XMMatrixTranslationFromVector(
    Offset: XMVECTOR,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = 1.0;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = 1.0;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = 1.0;
        M.m[2][3] = 0.0;

        M.m[3][0] = Offset.vector4_f32[0];
        M.m[3][1] = Offset.vector4_f32[1];
        M.m[3][2] = Offset.vector4_f32[2];
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(any(_XM_SSE_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = g_XMIdentityR0.v;
        M.r[1] = g_XMIdentityR1.v;
        M.r[2] = g_XMIdentityR2.v;
        M.r[3] = XMVectorSelect(g_XMIdentityR3.v, Offset, g_XMSelect1110.v);
        return M;
    }
}

/// Builds a matrix that scales along the x-axis, y-axis, and z-axis.
///
/// ## Parameters
///
/// `ScaleX` Scaling factor along the `x-axis`.
///
/// `ScaleY` Scaling factor along the `y-axis`.
///
/// `ScaleZ` Scaling factor along the `z-axis`.
///
/// ## Return value
///
/// Returns the scaling matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixScaling>
#[inline]
pub fn XMMatrixScaling(
    ScaleX: f32,
    ScaleY: f32,
    ScaleZ: f32,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = ScaleX;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = ScaleY;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = ScaleZ;
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = 0.0;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    unsafe {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = _mm_set_ps(0.0, 0.0, 0.0, ScaleX);
        M.r[1] = _mm_set_ps(0.0, 0.0, ScaleY, 0.0);
        M.r[2] = _mm_set_ps(0.0, ScaleZ, 0.0, 0.0);
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Builds a matrix that scales along the x-axis, y-axis, and z-axis.
///
/// ## Parameters
///
/// `Scale` 3D vector describing the scaling along the `x-axis`, `y-axis`, and `z-axis`.
///
/// ## Return value
///
/// Returns the scaling matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixScalingFromVector>
#[inline]
pub fn XMMatrixScalingFromVector(
    Scale: XMVECTOR,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = Scale.vector4_f32[0];
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = Scale.vector4_f32[1];
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = Scale.vector4_f32[2];
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = 0.0;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    unsafe {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = _mm_and_ps(Scale, *g_XMMaskX);
        M.r[1] = _mm_and_ps(Scale, *g_XMMaskY);
        M.r[2] = _mm_and_ps(Scale, *g_XMMaskZ);
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Builds a matrix that rotates around the `x-axis`.
///
/// ## Parameters
///
/// `Angle` Angle of rotation around the `x-axis`, in radians. Angles are measured clockwise when looking along the
/// rotation axis toward the origin.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationX>
#[inline]
pub fn XMMatrixRotationX(
    Angle: f32,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut fSinAngle: f32 = 0.0;
        let mut fCosAngle: f32 = 0.0;
        XMScalarSinCos(&mut fSinAngle, &mut fCosAngle, Angle);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = 1.0;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = fCosAngle;
        M.m[1][2] = fSinAngle;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = -fSinAngle;
        M.m[2][2] = fCosAngle;
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = 0.0;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut SinAngle: f32 = 0.0;
        let mut CosAngle: f32 = 0.0;
        XMScalarSinCos(&mut SinAngle, &mut CosAngle, Angle);

        let vSin: XMVECTOR = _mm_set_ss(SinAngle);
        let mut vCos: XMVECTOR = _mm_set_ss(CosAngle);
        // x = 0,y = cos,z = sin, w = 0
        vCos = _mm_shuffle_ps(vCos, vSin, _MM_SHUFFLE(3, 0, 0, 3));
        let mut M: XMMATRIX = undefined();
        M.r[0] = g_XMIdentityR0.v;
        M.r[1] = vCos;
        // x = 0,y = sin,z = cos, w = 0
        vCos = XM_PERMUTE_PS!(vCos, _MM_SHUFFLE(3, 1, 2, 0));
        // x = 0,y = -sin,z = cos, w = 0
        vCos = _mm_mul_ps(vCos, g_XMNegateY.v);
        M.r[2] = vCos;
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Builds a matrix that rotates around the `y-axis`.
///
/// ## Parameters
///
/// `Angle` Angle of rotation around the `y-axis`, in radians. Angles are measured clockwise when looking along
/// the rotation axis toward the origin.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationY>
#[inline]
pub fn XMMatrixRotationY(
    Angle: f32,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut fSinAngle: f32 = 0.0;
        let mut fCosAngle: f32 = 0.0;
        XMScalarSinCos(&mut fSinAngle, &mut fCosAngle, Angle);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = fCosAngle;
        M.m[0][1] = 0.0;
        M.m[0][2] = -fSinAngle;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = 1.0;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = fSinAngle;
        M.m[2][1] = 0.0;
        M.m[2][2] = fCosAngle;
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = 0.0;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut SinAngle: f32 = 0.0;
        let mut CosAngle: f32 = 0.0;
        XMScalarSinCos(&mut SinAngle, &mut CosAngle, Angle);

        let mut vSin: XMVECTOR = _mm_set_ss(SinAngle);
        let vCos: XMVECTOR = _mm_set_ss(CosAngle);
        // x = sin,y = 0,z = cos, w = 0
        vSin = _mm_shuffle_ps(vSin, vCos, _MM_SHUFFLE(3, 0, 3, 0));
        let mut M: XMMATRIX = undefined();
        M.r[2] = vSin;
        M.r[1] = g_XMIdentityR1.v;
        // x = cos,y = 0,z = sin, w = 0
        vSin = XM_PERMUTE_PS!(vSin, _MM_SHUFFLE(3, 0, 1, 2));
        // x = cos,y = 0,z = -sin, w = 0
        vSin = _mm_mul_ps(vSin, g_XMNegateZ.v);
        M.r[0] = vSin;
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Builds a matrix that rotates around the `z-axis`.
///
/// ## Parameters
///
/// `Angle` Angle of rotation around the `z-axis`, in radians. Angles are measured clockwise when looking along
/// the rotation axis toward the origin.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationZ>
#[inline]
pub fn XMMatrixRotationZ(
    Angle: f32,
) -> XMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut fSinAngle: f32 = 0.0;
        let mut fCosAngle: f32 = 0.0;
        XMScalarSinCos(&mut fSinAngle, &mut fCosAngle, Angle);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = fCosAngle;
        M.m[0][1] = fSinAngle;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = -fSinAngle;
        M.m[1][1] = fCosAngle;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = 1.0;
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = 0.0;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut SinAngle: f32 = 0.0;
        let mut CosAngle: f32 = 0.0;
        XMScalarSinCos(&mut SinAngle, &mut CosAngle, Angle);

        let vSin: XMVECTOR = _mm_set_ss(SinAngle);
        let mut vCos: XMVECTOR = _mm_set_ss(CosAngle);
        // x = cos,y = sin,z = 0, w = 0
        vCos = _mm_unpacklo_ps(vCos, vSin);
        let mut M: XMMATRIX = undefined();
        M.r[0] = vCos;
        // x = sin,y = cos,z = 0, w = 0
        vCos = XM_PERMUTE_PS!(vCos, _MM_SHUFFLE(3, 2, 0, 1));
        // x = cos,y = -sin,z = 0, w = 0
        vCos = _mm_mul_ps(vCos, g_XMNegateX.v);
        M.r[1] = vCos;
        M.r[2] = g_XMIdentityR2.v;
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Builds a rotation matrix based on a given `pitch`, `yaw`, and `roll` (Euler angles).
///
/// ## Parameters
///
/// `Pitch` Angle of rotation around the `x-axis`, in radians.
///
/// `Yaw` Angle of rotation around the `y-axis`, in radians.
///
/// `Roll` Angle of rotation around the `z-axis`, in radians.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Remarks
///
/// Angles are measured clockwise when looking along the rotation axis toward the origin. **This is a left-handed
/// coordinate system. To use right-handed coordinates, negate all three angles.**
///
/// The order of transformations is `roll` first, then `pitch`, and then `yaw`. The rotations are all applied
/// in the global coordinate frame.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationRollPitchYaw>
#[inline]
pub fn XMMatrixRotationRollPitchYaw(
    Pitch: f32,
    Yaw: f32,
    Roll: f32,
) -> XMMATRIX
{
    let Angles: XMVECTOR = XMVectorSet(Pitch, Yaw, Roll, 0.0);
    return XMMatrixRotationRollPitchYawFromVector(Angles);
}

/// Builds a rotation matrix based on a vector containing the Euler angles (`pitch`, `yaw`, and `roll`).
///
/// ## Parameters
///
/// `Angles` 3D vector containing the Euler angles in the order `pitch`, then `yaw`, and then `roll`.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Remarks
///
/// Angles are measured clockwise when looking along the rotation axis toward the origin. **This is a left-handed
/// coordinate system. To use right-handed coordinates, negate all three angles**.
///
/// The order of transformations is `roll` first, then `pitch`, and then `yaw`. The rotations are all applied
/// in the global coordinate frame.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationRollPitchYawFromVector>
#[inline]
pub fn XMMatrixRotationRollPitchYawFromVector(
    Angles: FXMVECTOR,
) -> XMMATRIX
{
    let Q: XMVECTOR = XMQuaternionRotationRollPitchYawFromVector(Angles);
    return XMMatrixRotationQuaternion(Q);
}

/// Builds a matrix that rotates around an arbitrary normal vector.
///
/// ## Parameters
///
/// `NormalAxis` Normal vector describing the axis of rotation.
///
/// `Angle` Angle of rotation in radians. Angles are measured clockwise when looking along the rotation axis toward
/// the origin.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationNormal>
#[inline]
pub fn XMMatrixRotationNormal(
    NormalAxis: FXMVECTOR,
    Angle: f32,
) -> XMMATRIX
{
    #[cfg(any(_XM_NO_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        let mut fSinAngle: f32 = 0.0;
        let mut fCosAngle: f32 = 0.0;
        XMScalarSinCos(&mut fSinAngle, &mut fCosAngle, Angle);

        let A: XMVECTOR = XMVectorSet(fSinAngle, fCosAngle, 1.0 - fCosAngle, 0.0);

        let C2: XMVECTOR = XMVectorSplatZ(A);
        let C1: XMVECTOR = XMVectorSplatY(A);
        let C0: XMVECTOR = XMVectorSplatX(A);

        let N0: XMVECTOR = <(XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_W)>::XMVectorSwizzle(NormalAxis);
        let N1: XMVECTOR = <(XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_W)>::XMVectorSwizzle(NormalAxis);

        let mut V0: XMVECTOR = XMVectorMultiply(C2, N0);
        V0 = XMVectorMultiply(V0, N1);

        let mut R0: XMVECTOR = XMVectorMultiply(C2, NormalAxis);
        R0 = XMVectorMultiplyAdd(R0, NormalAxis, C1);

        let R1: XMVECTOR = XMVectorMultiplyAdd(C0, NormalAxis, V0);
        let R2: XMVECTOR = XMVectorNegativeMultiplySubtract(C0, NormalAxis, V0);

        V0 = XMVectorSelect(A, R0, g_XMSelect1110.v);
        let V1: XMVECTOR = <(XM_PERMUTE_0Z, XM_PERMUTE_1Y, XM_PERMUTE_1Z, XM_PERMUTE_0X)>::XMVectorPermute(R1, R2);
        let V2: XMVECTOR = <(XM_PERMUTE_0Y, XM_PERMUTE_1X, XM_PERMUTE_0Y, XM_PERMUTE_1X)>::XMVectorPermute(R1, R2);

        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = <(XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1Y, XM_PERMUTE_0W)>::XMVectorPermute(V0, V1);
        M.r[1] = <(XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_1W, XM_PERMUTE_0W)>::XMVectorPermute(V0, V1);
        M.r[2] = <(XM_PERMUTE_1X, XM_PERMUTE_1Y, XM_PERMUTE_0Z, XM_PERMUTE_0W)>::XMVectorPermute(V0, V2);
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut fSinAngle: f32 = 0.0;
        let mut fCosAngle: f32 = 0.0;
        XMScalarSinCos(&mut fSinAngle, &mut fCosAngle, Angle);

        let C2: XMVECTOR = _mm_set_ps1(1.0 - fCosAngle);
        let C1: XMVECTOR = _mm_set_ps1(fCosAngle);
        let C0: XMVECTOR = _mm_set_ps1(fSinAngle);

        let N0: XMVECTOR = XM_PERMUTE_PS!(NormalAxis, _MM_SHUFFLE(3, 0, 2, 1));
        let N1: XMVECTOR = XM_PERMUTE_PS!(NormalAxis, _MM_SHUFFLE(3, 1, 0, 2));

        let mut V0: XMVECTOR = _mm_mul_ps(C2, N0);
        V0 = _mm_mul_ps(V0, N1);

        let mut R0: XMVECTOR = _mm_mul_ps(C2, NormalAxis);
        R0 = _mm_mul_ps(R0, NormalAxis);
        R0 = _mm_add_ps(R0, C1);

        let mut R1: XMVECTOR = _mm_mul_ps(C0, NormalAxis);
        R1 = _mm_add_ps(R1, V0);
        let mut R2: XMVECTOR = _mm_mul_ps(C0, NormalAxis);
        R2 = _mm_sub_ps(V0, R2);

        V0 = _mm_and_ps(R0, g_XMMask3.v);
        let mut V1: XMVECTOR = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(2, 1, 2, 0));
        V1 = XM_PERMUTE_PS!(V1, _MM_SHUFFLE(0, 3, 2, 1));
        let mut V2: XMVECTOR = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(0, 0, 1, 1));
        V2 = XM_PERMUTE_PS!(V2, _MM_SHUFFLE(2, 0, 2, 0));

        R2 = _mm_shuffle_ps(V0, V1, _MM_SHUFFLE(1, 0, 3, 0));
        R2 = XM_PERMUTE_PS!(R2, _MM_SHUFFLE(1, 3, 2, 0));

        let mut M: XMMATRIX = undefined();
        M.r[0] = R2;

        R2 = _mm_shuffle_ps(V0, V1, _MM_SHUFFLE(3, 2, 3, 1));
        R2 = XM_PERMUTE_PS!(R2, _MM_SHUFFLE(1, 3, 0, 2));
        M.r[1] = R2;

        V2 = _mm_shuffle_ps(V2, V0, _MM_SHUFFLE(3, 2, 1, 0));
        M.r[2] = V2;
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }
}

/// Builds a matrix that rotates around an arbitrary axis.
///
/// ## Parameters
///
/// `Axis` Vector describing the axis of rotation.
///
/// `Angle` Angle of rotation in radians. Angles are measured clockwise when looking along the rotation axis toward
/// the origin.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Remarks
///
/// If Axis is a normalized vector, it is faster to use the [`XMMatrixRotationNormal`] function to build this
/// type of matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationAxis>
///
/// [`XMMatrixRotationNormal`]: crate::matrix::XMMatrixRotationNormal
#[inline]
pub fn XMMatrixRotationAxis(
    Axis: FXMVECTOR,
    Angle: f32,
) -> XMMATRIX
{
    debug_assert!(!XMVector3Equal(Axis, XMVectorZero()));
    debug_assert!(!XMVector3IsInfinite(Axis));

    let Normal: XMVECTOR = XMVector3Normalize(Axis);
    return XMMatrixRotationNormal(Normal, Angle);
}

/// Builds a rotation matrix from a quaternion.
///
/// ## Parameters
///
/// `Quaternion` Quaternion defining the rotation.
///
/// ## Return value
///
/// Returns the rotation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixRotationQuaternion>
#[inline]
pub fn XMMatrixRotationQuaternion(
    Quaternion: FXMVECTOR ,
) -> XMMATRIX
{
    #[cfg(any(_XM_NO_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        // PERFORMANCE: static const
        const Constant1110: XMVECTORF32 = XMVECTORF32 { f: [ 1.0, 1.0, 1.0, 0.0 ] };

        let Q0: XMVECTOR = XMVectorAdd(Quaternion, Quaternion);
        let Q1: XMVECTOR = XMVectorMultiply(Quaternion, Q0);

        let mut V0: XMVECTOR = <(XM_PERMUTE_0Y, XM_PERMUTE_0X, XM_PERMUTE_0X, XM_PERMUTE_1W)>::XMVectorPermute(Q1, Constant1110.v);
        let mut V1: XMVECTOR = <(XM_PERMUTE_0Z, XM_PERMUTE_0Z, XM_PERMUTE_0Y, XM_PERMUTE_1W)>::XMVectorPermute(Q1, Constant1110.v);
        let mut R0: XMVECTOR = XMVectorSubtract(*Constant1110, V0);
        R0 = XMVectorSubtract(R0, V1);

        V0 = <(XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_W)>::XMVectorSwizzle(Quaternion);
        V1 = <(XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_W)>::XMVectorSwizzle(Q0);
        V0 = XMVectorMultiply(V0, V1);

        V1 = XMVectorSplatW(Quaternion);
        let V2: XMVECTOR = <(XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X, XM_SWIZZLE_W)>::XMVectorSwizzle(Q0);
        V1 = XMVectorMultiply(V1, V2);

        let R1: XMVECTOR = XMVectorAdd(V0, V1);
        let R2: XMVECTOR = XMVectorSubtract(V0, V1);

        V0 = <(XM_PERMUTE_0Y, XM_PERMUTE_1X, XM_PERMUTE_1Y, XM_PERMUTE_0Z)>::XMVectorPermute(R1, R2);
        V1 = <(XM_PERMUTE_0X, XM_PERMUTE_1Z, XM_PERMUTE_0X, XM_PERMUTE_1Z)>::XMVectorPermute(R1, R2);

        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = <(XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1Y, XM_PERMUTE_0W)>::XMVectorPermute(R0, V0);
        M.r[1] = <(XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_1W, XM_PERMUTE_0W)>::XMVectorPermute(R0, V0);
        M.r[2] = <(XM_PERMUTE_1X, XM_PERMUTE_1Y, XM_PERMUTE_0Z, XM_PERMUTE_0W)>::XMVectorPermute(R0, V1);
        M.r[3] = g_XMIdentityR3.v;
        return M;
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // PERFORMANCE: static const
        const Constant1110: XMVECTORF32 = XMVECTORF32 { f: [ 1.0, 1.0, 1.0, 0.0 ] };

        let Q0: XMVECTOR = _mm_add_ps(Quaternion, Quaternion);
        let mut Q1: XMVECTOR = _mm_mul_ps(Quaternion, Q0);

        let mut V0: XMVECTOR = XM_PERMUTE_PS!(Q1, _MM_SHUFFLE(3, 0, 0, 1));
        V0 = _mm_and_ps(V0, *g_XMMask3);
        let mut V1: XMVECTOR = XM_PERMUTE_PS!(Q1, _MM_SHUFFLE(3, 1, 2, 2));
        V1 = _mm_and_ps(V1, *g_XMMask3);
        let mut R0: XMVECTOR = _mm_sub_ps(*Constant1110, V0);
        R0 = _mm_sub_ps(R0, V1);

        V0 = XM_PERMUTE_PS!(Quaternion, _MM_SHUFFLE(3, 1, 0, 0));
        V1 = XM_PERMUTE_PS!(Q0, _MM_SHUFFLE(3, 2, 1, 2));
        V0 = _mm_mul_ps(V0, V1);

        V1 = XM_PERMUTE_PS!(Quaternion, _MM_SHUFFLE(3, 3, 3, 3));
        let V2: XMVECTOR = XM_PERMUTE_PS!(Q0, _MM_SHUFFLE(3, 0, 2, 1));
        V1 = _mm_mul_ps(V1, V2);

        let R1: XMVECTOR = _mm_add_ps(V0, V1);
        let R2: XMVECTOR = _mm_sub_ps(V0, V1);

        V0 = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(1, 0, 2, 1));
        V0 = XM_PERMUTE_PS!(V0, _MM_SHUFFLE(1, 3, 2, 0));
        V1 = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(2, 2, 0, 0));
        V1 = XM_PERMUTE_PS!(V1, _MM_SHUFFLE(2, 0, 2, 0));

        Q1 = _mm_shuffle_ps(R0, V0, _MM_SHUFFLE(1, 0, 3, 0));
        Q1 = XM_PERMUTE_PS!(Q1, _MM_SHUFFLE(1, 3, 2, 0));

        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = Q1;

        Q1 = _mm_shuffle_ps(R0, V0, _MM_SHUFFLE(3, 2, 3, 1));
        Q1 = XM_PERMUTE_PS!(Q1, _MM_SHUFFLE(1, 3, 0, 2));
        M.r[1] = Q1;

        Q1 = _mm_shuffle_ps(V1, R0, _MM_SHUFFLE(3, 2, 1, 0));
        M.r[2] = Q1;
        M.r[3] = *g_XMIdentityR3;
        return M;
    }
}

/// Builds a 2D transformation matrix.
///
/// ## Parameters
///
/// `ScalingOrigin` 2D vector describing the center of the scaling.
///
/// `ScalingOrientation` Scaling rotation factor.
///
/// `Scaling` 2D vector containing the scaling factors for the `x-axis` and `y-axis`.
///
/// `RotationOrigin` 2D vector describing the center of the rotation.
///
/// `Rotation` Angle of rotation, in radians.
///
/// `Translation` 2D vector describing the translation.
///
/// ## Return value
///
/// Returns the transformation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixTransformation2D>
#[inline]
pub fn XMMatrixTransformation2D(
    ScalingOrigin: FXMVECTOR,
    ScalingOrientation: f32,
    Scaling: FXMVECTOR,
    RotationOrigin: FXMVECTOR,
    Rotation: f32,
    Translation: GXMVECTOR
) -> XMMATRIX
{
    unsafe {
        // M = Inverse(MScalingOrigin) * Transpose(MScalingOrientation) * MScaling * MScalingOrientation *
        //         MScalingOrigin * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;

        let VScalingOrigin: XMVECTOR = XMVectorSelect(g_XMSelect1100.v, ScalingOrigin, g_XMSelect1100.v);
        let NegScalingOrigin: XMVECTOR = XMVectorNegate(VScalingOrigin);

        let MScalingOriginI: XMMATRIX = XMMatrixTranslationFromVector(NegScalingOrigin);
        let MScalingOrientation: XMMATRIX = XMMatrixRotationZ(ScalingOrientation);
        let MScalingOrientationT: XMMATRIX = XMMatrixTranspose(MScalingOrientation);
        let VScaling: XMVECTOR = XMVectorSelect(g_XMOne.v, Scaling, g_XMSelect1100.v);
        let MScaling: XMMATRIX = XMMatrixScalingFromVector(VScaling);
        let VRotationOrigin: XMVECTOR = XMVectorSelect(g_XMSelect1100.v, RotationOrigin, g_XMSelect1100.v);
        let MRotation: XMMATRIX = XMMatrixRotationZ(Rotation);
        let VTranslation: XMVECTOR = XMVectorSelect(g_XMSelect1100.v, Translation, g_XMSelect1100.v);

        let mut M: XMMATRIX = XMMatrixMultiply(MScalingOriginI, &MScalingOrientationT);
        M = XMMatrixMultiply(M, &MScaling);
        M = XMMatrixMultiply(M, &MScalingOrientation);
        M.r[3] = XMVectorAdd(M.r[3], VScalingOrigin);
        M.r[3] = XMVectorSubtract(M.r[3], VRotationOrigin);
        M = XMMatrixMultiply(M, &MRotation);
        M.r[3] = XMVectorAdd(M.r[3], VRotationOrigin);
        M.r[3] = XMVectorAdd(M.r[3], VTranslation);

        return M;
    }
}

/// Builds a transformation matrix.
///
/// ## Parameters
///
/// `ScalingOrigin` 3D vector describing the center of the scaling.
///
/// `ScalingOrientationQuaternion` Quaternion describing the orientation of the scaling.
///
/// `Scaling` 3D vector containing the scaling factors for the `x-axis`, `y-axis`, and `z-axis`.
///
/// `RotationOrigin` 3D vector describing the center of the rotation.
///
/// `RotationQuaternion` Quaternion describing the rotation around the origin indicated by RotationOrigin.
///
/// `Translation` 3D vector describing the translations along the `x-axis`, `y-axis`, and `z-axis`.
///
/// ## Return value
///
/// Returns the transformation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixTransformation>
#[inline]
pub fn XMMatrixTransformation(
    ScalingOrigin: FXMVECTOR,
    ScalingOrientationQuaternion: FXMVECTOR,
    Scaling: FXMVECTOR,
    RotationOrigin: GXMVECTOR,
    RotationQuaternion: HXMVECTOR,
    Translation: HXMVECTOR
) -> XMMATRIX
{
    unsafe {
        // M = Inverse(MScalingOrigin) * Transpose(MScalingOrientation) * MScaling * MScalingOrientation *
        //         MScalingOrigin * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;

        let VScalingOrigin: XMVECTOR = XMVectorSelect(g_XMSelect1110.v, ScalingOrigin, g_XMSelect1110.v);
        let NegScalingOrigin: XMVECTOR = XMVectorNegate(ScalingOrigin);

        let MScalingOriginI: XMMATRIX = XMMatrixTranslationFromVector(NegScalingOrigin);
        let MScalingOrientation: XMMATRIX = XMMatrixRotationQuaternion(ScalingOrientationQuaternion);
        let MScalingOrientationT: XMMATRIX = XMMatrixTranspose(MScalingOrientation);
        let MScaling: XMMATRIX = XMMatrixScalingFromVector(Scaling);
        let VRotationOrigin: XMVECTOR = XMVectorSelect(g_XMSelect1110.v, RotationOrigin, g_XMSelect1110.v);
        let MRotation: XMMATRIX = XMMatrixRotationQuaternion(RotationQuaternion);
        let VTranslation: XMVECTOR = XMVectorSelect(g_XMSelect1110.v, Translation, g_XMSelect1110.v);

        let mut M: XMMATRIX;
        M = XMMatrixMultiply(MScalingOriginI, &MScalingOrientationT);
        M = XMMatrixMultiply(M, &MScaling);
        M = XMMatrixMultiply(M, &MScalingOrientation);
        M.r[3] = XMVectorAdd(M.r[3], VScalingOrigin);
        M.r[3] = XMVectorSubtract(M.r[3], VRotationOrigin);
        M = XMMatrixMultiply(M, &MRotation);
        M.r[3] = XMVectorAdd(M.r[3], VRotationOrigin);
        M.r[3] = XMVectorAdd(M.r[3], VTranslation);
        return M;
    }
}

/// Builds a 2D affine transformation matrix in the xy plane.
///
/// ## Parameters
///
/// `Scaling` 2D vector of scaling factors for the x-coordinate and y-coordinate.
///
/// `RotationOrigin` 2D vector describing the center of rotation.
///
/// `Rotation` Radian angle of rotation.
///
/// `Translation` 2D vector translation offsets.
///
/// ## Return value
///
/// Returns the 2D affine transformation matrix.
///
/// ## Reference
//
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixAffineTransformation2D>
#[inline]
pub fn XMMatrixAffineTransformation2D(
    Scaling: FXMVECTOR,
    RotationOrigin: FXMVECTOR,
    Rotation: f32,
    Translation: GXMVECTOR
) -> XMMATRIX
{
    unsafe {
        // M = MScaling * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;

        let VScaling: XMVECTOR = XMVectorSelect(g_XMOne.v, Scaling, g_XMSelect1100.v);
        let MScaling: XMMATRIX = XMMatrixScalingFromVector(VScaling);
        let VRotationOrigin: XMVECTOR = XMVectorSelect(g_XMSelect1100.v, RotationOrigin, g_XMSelect1100.v);
        let MRotation: XMMATRIX = XMMatrixRotationZ(Rotation);
        let VTranslation: XMVECTOR = XMVectorSelect(g_XMSelect1100.v, Translation, g_XMSelect1100.v);

        let mut M: XMMATRIX;
        M = MScaling;
        M.r[3] = XMVectorSubtract(M.r[3], VRotationOrigin);
        M = XMMatrixMultiply(M, &MRotation);
        M.r[3] = XMVectorAdd(M.r[3], VRotationOrigin);
        M.r[3] = XMVectorAdd(M.r[3], VTranslation);
        return M;
    }
}

/// Builds an affine transformation matrix.
///
/// ## Parameters
///
/// `Scaling` Vector of scaling factors for each dimension.
///
/// `RotationOrigin` Point identifying the center of rotation.
///
/// `RotationQuaternion` Rotation factors.
///
/// `Translation` Translation offsets.
///
/// ## Return value
///
/// Returns the affine transformation matrix, built from the scaling, rotation, and translation information.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixAffineTransformation>
#[inline]
pub fn XMMatrixAffineTransformation(
    Scaling: FXMVECTOR,
    RotationOrigin: FXMVECTOR,
    RotationQuaternion: FXMVECTOR,
    Translation: GXMVECTOR
) -> XMMATRIX
{
    unsafe {
        // M = MScaling * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;

        let MScaling: XMMATRIX = XMMatrixScalingFromVector(Scaling);
        let VRotationOrigin: XMVECTOR = XMVectorSelect(g_XMSelect1110.v, RotationOrigin, g_XMSelect1110.v);
        let MRotation: XMMATRIX = XMMatrixRotationQuaternion(RotationQuaternion);
        let VTranslation: XMVECTOR = XMVectorSelect(g_XMSelect1110.v, Translation, g_XMSelect1110.v);

        let mut M: XMMATRIX;
        M = MScaling;
        M.r[3] = XMVectorSubtract(M.r[3], VRotationOrigin);
        M = XMMatrixMultiply(M, &MRotation);
        M.r[3] = XMVectorAdd(M.r[3], VRotationOrigin);
        M.r[3] = XMVectorAdd(M.r[3], VTranslation);
        return M;
    }
}

/// Builds a transformation matrix designed to reflect vectors through a given plane.
///
/// ## Parameters
///
/// `ReflectionPlane` Plane to reflect through.
///
/// ## Return value
///
/// Returns the transformation matrix.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixReflect>
#[inline]
pub fn XMMatrixReflect(
    ReflectionPlane: FXMVECTOR
) -> XMMATRIX
{
    unsafe {
        debug_assert!(!XMVector3Equal(ReflectionPlane, XMVectorZero()));
        debug_assert!(!XMPlaneIsInfinite(ReflectionPlane));

        const NegativeTwo: XMVECTORF32 = XMVECTORF32 { f: [ -2.0, -2.0, -2.0, 0.0 ] };

        let P: XMVECTOR = XMPlaneNormalize(ReflectionPlane);
        let S: XMVECTOR = XMVectorMultiply(P, NegativeTwo.v);

        let A: XMVECTOR = XMVectorSplatX(P);
        let B: XMVECTOR = XMVectorSplatY(P);
        let C: XMVECTOR = XMVectorSplatZ(P);
        let D: XMVECTOR = XMVectorSplatW(P);

        let mut M: XMMATRIX = undefined();
        M.r[0] = XMVectorMultiplyAdd(A, S, g_XMIdentityR0.v);
        M.r[1] = XMVectorMultiplyAdd(B, S, g_XMIdentityR1.v);
        M.r[2] = XMVectorMultiplyAdd(C, S, g_XMIdentityR2.v);
        M.r[3] = XMVectorMultiplyAdd(D, S, g_XMIdentityR3.v);
        return M;
    }
}

/// Builds a transformation matrix that flattens geometry into a plane.
///
/// ## Parameters
///
/// `ShadowPlane` Reference plane.
///
/// `LightPosition` 4D vector describing the light's position. If the light's w-component is `0.0`, the ray from the origin
/// to the light represents a directional light. If it is `1.0`, the light is a point light.
///
/// ## Return value
///
/// Returns the transformation matrix that flattens the geometry into the plane ShadowPlane.
///
/// ## Remarks
///
/// This function is useful for forming planar-projected shadows from a light source.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixShadow>
#[inline]
pub fn XMMatrixShadow(
    ShadowPlane: FXMVECTOR,
    LightPosition: FXMVECTOR,
) -> XMMATRIX
{
    unsafe {
        const Select0001: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_0, XM_SELECT_0, XM_SELECT_1 ] };

        debug_assert!(!XMVector3Equal(ShadowPlane, XMVectorZero()));
        debug_assert!(!XMPlaneIsInfinite(ShadowPlane));

        let mut P: XMVECTOR = XMPlaneNormalize(ShadowPlane);
        let mut Dot: XMVECTOR = XMPlaneDot(P, LightPosition);
        P = XMVectorNegate(P);
        let D: XMVECTOR = XMVectorSplatW(P);
        let C: XMVECTOR = XMVectorSplatZ(P);
        let B: XMVECTOR = XMVectorSplatY(P);
        let A: XMVECTOR = XMVectorSplatX(P);
        Dot = XMVectorSelect(Select0001.v, Dot, Select0001.v);

        let mut M: XMMATRIX = undefined();
        M.r[3] = XMVectorMultiplyAdd(D, LightPosition, Dot);
        Dot = XMVectorRotateLeft(Dot, 1);
        M.r[2] = XMVectorMultiplyAdd(C, LightPosition, Dot);
        Dot = XMVectorRotateLeft(Dot, 1);
        M.r[1] = XMVectorMultiplyAdd(B, LightPosition, Dot);
        Dot = XMVectorRotateLeft(Dot, 1);
        M.r[0] = XMVectorMultiplyAdd(A, LightPosition, Dot);
        return M;
    }
}

/// Builds a view matrix for a left-handed coordinate system using a camera position, an up direction, and a focal point.
///
/// ## Parameters
///
/// `EyePosition` Position of the camera.
///
/// `FocusPosition` Position of the focal point.
///
/// `UpDirection` Up direction of the camera, typically < `0.0`, `1.0`, `0.0` >.
///
/// ## Return value
///
/// Returns a view matrix that transforms a point from world space into view space.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixLookAtLH>
#[inline]
pub fn XMMatrixLookAtLH(
    EyePosition: FXMVECTOR,
    FocusPosition: FXMVECTOR,
    UpDirection: FXMVECTOR,
) -> XMMATRIX
{
    let EyeDirection: XMVECTOR = XMVectorSubtract(FocusPosition, EyePosition);
    return XMMatrixLookToLH(EyePosition, EyeDirection, UpDirection);
}

/// Builds a view matrix for a right-handed coordinate system using a camera position, an up direction, and a focal point.
///
/// ## Parameters
///
/// `EyePosition` Position of the camera.
///
/// `FocusPosition` Position of the focal point.
///
/// `UpDirection` Up direction of the camera, typically < `0.0`, `1.0`, `0.0` >.
///
/// ## Return value
///
/// Returns a view matrix that transforms a point from world space into view space.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixLookAtRH>
#[inline]
pub fn XMMatrixLookAtRH(
    EyePosition: FXMVECTOR,
    FocusPosition: FXMVECTOR,
    UpDirection: FXMVECTOR,
) -> XMMATRIX
{
    let NegEyeDirection: XMVECTOR = XMVectorSubtract(EyePosition, FocusPosition);
    return XMMatrixLookToLH(EyePosition, NegEyeDirection, UpDirection);
}

/// Builds a view matrix for a left-handed coordinate system using a camera position, an up direction, and a camera direction.
///
/// ## Parameters
///
/// `EyePosition` Position of the camera.
///
/// `EyeDirection` Direction of the camera.
///
/// `UpDirection` Up direction of the camera, typically < `0.0`, `1.0`, `0.0` >.
///
/// ## Return value
///
/// Returns a view matrix that transforms a point from world space into view space.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixLookToLH>
#[inline]
pub fn XMMatrixLookToLH(
    EyePosition: FXMVECTOR,
    EyeDirection: FXMVECTOR,
    UpDirection: FXMVECTOR,
) -> XMMATRIX
{
    unsafe {
        debug_assert!(!XMVector3Equal(EyeDirection, XMVectorZero()));
        debug_assert!(!XMVector3IsInfinite(EyeDirection));
        debug_assert!(!XMVector3Equal(UpDirection, XMVectorZero()));
        debug_assert!(!XMVector3IsInfinite(UpDirection));

        let R2: XMVECTOR = XMVector3Normalize(EyeDirection);

        let mut R0: XMVECTOR = XMVector3Cross(UpDirection, R2);
        R0 = XMVector3Normalize(R0);

        let R1: XMVECTOR = XMVector3Cross(R2, R0);

        let NegEyePosition: XMVECTOR = XMVectorNegate(EyePosition);

        let D0: XMVECTOR = XMVector3Dot(R0, NegEyePosition);
        let D1: XMVECTOR = XMVector3Dot(R1, NegEyePosition);
        let D2: XMVECTOR = XMVector3Dot(R2, NegEyePosition);

        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = XMVectorSelect(D0, R0, g_XMSelect1110.v);
        M.r[1] = XMVectorSelect(D1, R1, g_XMSelect1110.v);
        M.r[2] = XMVectorSelect(D2, R2, g_XMSelect1110.v);
        M.r[3] = g_XMIdentityR3.v;

        M = XMMatrixTranspose(M);

        return M;
    }
}

/// Builds a view matrix for a right-handed coordinate system using a camera position, an up direction, and a camera direction.
///
/// ## Parameters
///
/// `EyePosition` Position of the camera.
///
/// `EyeDirection` Direction of the camera.
///
/// `UpDirection` Up direction of the camera, typically < `0.0`, `1.0`, `0.0` >.
///
/// ## Return value
///
/// Returns a view matrix that transforms a point from world space into view space.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixLookToRH>
#[inline]
pub fn XMMatrixLookToRH(
    EyePosition: FXMVECTOR,
    EyeDirection: FXMVECTOR,
    UpDirection: FXMVECTOR,
) -> XMMATRIX
{
    let NegEyeDirection: XMVECTOR = XMVectorNegate(EyeDirection);
    return XMMatrixLookToLH(EyePosition, NegEyeDirection, UpDirection);
}

/// Builds a left-handed perspective projection matrix.
///
/// ## Parameters
///
/// `ViewWidth` Width of the frustum at the near clipping plane.
///
/// `ViewHeight` Height of the frustum at the near clipping plane.
///
/// `NearZ` Distance to the near clipping plane. Must be greater than zero.
///
/// `FarZ` Distance to the far clipping plane. Must be greater than zero.
///
/// ## Return value
///
/// Returns the perspective projection matrix.
///
/// ## Remarks
///
/// For typical usage, `NearZ` is less than `FarZ`. However, if you flip these values so `FarZ` is less than `NearZ`,
/// the result is an inverted `z` buffer which can provide increased floating-point precision. `NearZ` and `FarZ`
/// cannot be the same value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixPerspectiveLH>
#[inline]
pub fn XMMatrixPerspectiveLH(
    ViewWidth: f32,
    ViewHeight: f32,
    NearZ: f32,
    FarZ: f32
) -> XMMATRIX
{
    debug_assert!(NearZ > 0.0 && FarZ > 0.0);
    debug_assert!(!XMScalarNearEqual(ViewWidth, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(ViewHeight, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let TwoNearZ: f32 = NearZ + NearZ;
        let fRange: f32 = FarZ / (FarZ - NearZ);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = TwoNearZ / ViewWidth;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = TwoNearZ / ViewHeight;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = fRange;
        M.m[2][3] = 1.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = -fRange * NearZ;
        M.m[3][3] = 0.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        let TwoNearZ: f32 = NearZ + NearZ;
        let fRange: f32 = FarZ / (FarZ - NearZ);
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            TwoNearZ / ViewWidth,
            TwoNearZ / ViewHeight,
            fRange,
            -fRange * NearZ
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // TwoNearZ / ViewWidth,0,0,0
        M.r[0] = vTemp;
        // 0,TwoNearZ / ViewHeight,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // x=fRange,y=-fRange * NearZ,0,1.0f
        vValues = _mm_shuffle_ps(vValues, *g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
        // 0,0,fRange,1.0f
        vTemp = _mm_setzero_ps();
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 0, 0, 0));
        M.r[2] = vTemp;
        // 0,0,-fRange * NearZ,0
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 1, 0, 0));
        M.r[3] = vTemp;
        return M;
    }
}

/// Builds a right-handed perspective projection matrix.
///
/// ## Parameters
///
/// `ViewWidth` Width of the frustum at the near clipping plane.
///
/// `ViewHeight` Height of the frustum at the near clipping plane.
///
/// `NearZ` Distance to the near clipping plane. Must be greater than zero.
///
/// `FarZ` Distance to the far clipping plane. Must be greater than zero.
///
/// ## Return value
///
/// Returns the perspective projection matrix.
///
/// ## Remarks
///
/// For typical usage, `NearZ` is less than `FarZ`. However, if you flip these values so `FarZ` is less than `NearZ`,
/// the result is an inverted `z` buffer which can provide increased floating-point precision. `NearZ` and `FarZ`
/// cannot be the same value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixPerspectiveRH>
#[inline]
pub fn XMMatrixPerspectiveRH(
    ViewWidth: f32,
    ViewHeight: f32,
    NearZ: f32,
    FarZ: f32
) -> XMMATRIX
{
    debug_assert!(NearZ > 0.0 && FarZ > 0.0);
    debug_assert!(!XMScalarNearEqual(ViewWidth, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(ViewHeight, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let TwoNearZ: f32 = NearZ + NearZ;
        let fRange: f32 = FarZ / (NearZ - FarZ);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = TwoNearZ / ViewWidth;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = TwoNearZ / ViewHeight;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = fRange;
        M.m[2][3] = -1.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = fRange * NearZ;
        M.m[3][3] = 0.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        let TwoNearZ: f32 = NearZ + NearZ;
        let fRange: f32 = FarZ / (NearZ - FarZ);
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            TwoNearZ / ViewWidth,
            TwoNearZ / ViewHeight,
            fRange,
            fRange * NearZ
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // TwoNearZ / ViewWidth,0,0,0
        M.r[0] = vTemp;
        // 0,TwoNearZ / ViewHeight,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // x=fRange,y=-fRange * NearZ,0,-1.0f
        vValues = _mm_shuffle_ps(vValues, *g_XMNegIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
        // 0,0,fRange,-1.0f
        vTemp = _mm_setzero_ps();
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 0, 0, 0));
        M.r[2] = vTemp;
        // 0,0,-fRange * NearZ,0
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 1, 0, 0));
        M.r[3] = vTemp;
        return M;
    }
}

/// Builds a left-handed perspective projection matrix based on a field of view.
///
/// ## Parameters
///
/// `FovAngleY` Top-down field-of-view angle in radians.
///
/// `AspectRatio` Aspect ratio of view-space `X:Y`.
///
/// `NearZ` Distance to the near clipping plane. Must be greater than zero.
///
/// `FarZ` Distance to the far clipping plane. Must be greater than zero.
///
/// ## Return value
///
/// Returns the perspective projection matrix.
///
/// ## Remarks
///
/// For typical usage, `NearZ` is less than `FarZ`. However, if you flip these values so `FarZ` is less than `NearZ`,
/// the result is an inverted `z` buffer which can provide increased floating-point precision. `NearZ` and `FarZ`
/// cannot be the same value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixPerspectiveFovLH>
#[inline]
pub fn XMMatrixPerspectiveFovLH(
    FovAngleY: f32,
    AspectRatio: f32,
    NearZ: f32,
    FarZ: f32
) -> XMMATRIX
{
    debug_assert!(NearZ > 0.0 && FarZ > 0.0);
    debug_assert!(!XMScalarNearEqual(FovAngleY, 0.0, 0.00001 * 2.0));
    debug_assert!(!XMScalarNearEqual(AspectRatio, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut SinFov: f32 = 0.0;
        let mut CosFov: f32 = 0.0;
        XMScalarSinCos(&mut SinFov, &mut CosFov, 0.5 * FovAngleY);

        let Height: f32 = CosFov / SinFov;
        let Width: f32 = Height / AspectRatio;
        let fRange: f32 = FarZ / (FarZ - NearZ);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = Width;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = Height;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = fRange;
        M.m[2][3] = 1.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = -fRange * NearZ;
        M.m[3][3] = 0.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut SinFov: f32 = 0.0;
        let mut CosFov: f32 = 0.0;
        XMScalarSinCos(&mut SinFov, &mut CosFov, 0.5 * FovAngleY);

        let fRange: f32 = FarZ / (FarZ - NearZ);
        // Note: This is recorded on the stack
        let Height: f32 = CosFov / SinFov;
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            Height / AspectRatio,
            Height,
            fRange,
            -fRange * NearZ
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // CosFov / SinFov,0,0,0
        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = vTemp;
        // 0,Height / AspectRatio,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // x=fRange,y=-fRange * NearZ,0,1.0f
        vTemp = _mm_setzero_ps();
        vValues = _mm_shuffle_ps(vValues, *g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
        // 0,0,fRange,1.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 0, 0, 0));
        M.r[2] = vTemp;
        // 0,0,-fRange * NearZ,0.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 1, 0, 0));
        M.r[3] = vTemp;
        return M;
    }
}


/// Builds a right-handed perspective projection matrix based on a field of view.
///
/// ## Parameters
///
/// `FovAngleY` Top-down field-of-view angle in radians.
///
/// `AspectRatio` Aspect ratio of view-space `X:Y`.
///
/// `NearZ` Distance to the near clipping plane. Must be greater than zero.
///
/// `FarZ` Distance to the far clipping plane. Must be greater than zero.
///
/// ## Return value
///
/// Returns the perspective projection matrix.
///
/// ## Remarks
///
/// For typical usage, `NearZ` is less than `FarZ`. However, if you flip these values so `FarZ` is less than `NearZ`,
/// the result is an inverted `z` buffer which can provide increased floating-point precision. `NearZ` and `FarZ`
/// cannot be the same value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixPerspectiveFovRH>
#[inline]
pub fn XMMatrixPerspectiveFovRH(
    FovAngleY: f32,
    AspectRatio: f32,
    NearZ: f32,
    FarZ: f32
) -> XMMATRIX
{
    debug_assert!(NearZ > 0.0 && FarZ > 0.0);
    debug_assert!(!XMScalarNearEqual(FovAngleY, 0.0, 0.00001 * 2.0));
    debug_assert!(!XMScalarNearEqual(AspectRatio, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut SinFov: f32 = 0.0;
        let mut CosFov: f32 = 0.0;
        XMScalarSinCos(&mut SinFov, &mut CosFov, 0.5 * FovAngleY);

        let Height: f32 = CosFov / SinFov;
        let Width: f32 = Height / AspectRatio;
        let fRange: f32 = FarZ / (NearZ - FarZ);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = Width;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = Height;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = fRange;
        M.m[2][3] = -1.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = fRange * NearZ;
        M.m[3][3] = 0.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut SinFov: f32 = 0.0;
        let mut CosFov: f32 = 0.0;
        XMScalarSinCos(&mut SinFov, &mut CosFov, 0.5 * FovAngleY);
        let fRange: f32 = FarZ / (NearZ - FarZ);
        let Height: f32 = CosFov / SinFov;
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            Height / AspectRatio,
            Height,
            fRange,
            fRange * NearZ
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // CosFov / SinFov,0,0,0
        let mut M: XMMATRIX = crate::undefined();
        M.r[0] = vTemp;
        // 0,Height / AspectRatio,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // x=fRange,y=-fRange * NearZ,0,-1.0f
        vTemp = _mm_setzero_ps();
        vValues = _mm_shuffle_ps(vValues, *g_XMNegIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
        // 0,0,fRange,-1.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 0, 0, 0));
        M.r[2] = vTemp;
        // 0,0,fRange * NearZ,0.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 1, 0, 0));
        M.r[3] = vTemp;
        return M;
    }
}

/// Builds a custom version of a left-handed perspective projection matrix.
/// 
/// ## Parameters
/// 
/// `ViewLeft` The x-coordinate of the left side of the clipping frustum at the near clipping plane.
/// 
/// `ViewRight` The x-coordinate of the right side of the clipping frustum at the near clipping plane.
/// 
/// `ViewBottom` The y-coordinate of the bottom side of the clipping frustum at the near clipping plane.
/// 
/// `ViewTop` The y-coordinate of the top side of the clipping frustum at the near clipping plane.
/// 
/// `NearZ` Distance to the near clipping plane. Must be greater than zero.
/// 
/// `FarZ` Distance to the far clipping plane. Must be greater than zero.
/// 
/// ## Return Values
/// 
/// Returns the custom perspective projection matrix.
/// 
/// ## Remarks
/// 
/// For typical usage, NearZ is less than FarZ. However, if you flip these values so FarZ is less than NearZ, 
/// the result is an inverted z buffer (also known as a "reverse z buffer") which can provide increased floating-point precision.
///
/// NearZ and FarZ cannot be the same value and must be greater than 0.
/// 
/// ## Reference
/// 
/// <https://learn.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-xmmatrixperspectiveoffcenterlh>
#[inline]
pub fn XMMatrixPerspectiveOffCenterLH(
    ViewLeft: f32,
    ViewRight: f32,
    ViewBottom: f32,
    ViewTop: f32,
    NearZ: f32,
    FarZ: f32,
) -> XMMATRIX {
    debug_assert!(NearZ > 0.0 && FarZ > 0.0);
    debug_assert!(!XMScalarNearEqual(ViewRight, ViewLeft, 0.00001));
    debug_assert!(!XMScalarNearEqual(ViewTop, ViewBottom, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let TwoNearZ = NearZ + NearZ;
        let ReciprocalWidth = 1.0 / (ViewRight - ViewLeft);
        let ReciprocalHeight = 1.0 / (ViewTop - ViewBottom);
        let fRange = FarZ / (FarZ - NearZ);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = TwoNearZ * ReciprocalWidth;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = TwoNearZ * ReciprocalHeight;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;
        
        M.m[2][0] = -(ViewLeft + ViewRight) * ReciprocalWidth;
        M.m[2][1] = -(ViewTop + ViewBottom) * ReciprocalHeight;
        M.m[2][2] = fRange;
        M.m[2][3] = 1.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = -fRange * NearZ;
        M.m[3][3] = 0.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        let TwoNearZ = NearZ + NearZ;
        let ReciprocalWidth = 1.0 / (ViewRight - ViewLeft);
        let ReciprocalHeight = 1.0 / (ViewTop - ViewBottom);
        let fRange = FarZ / (FarZ - NearZ);
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            TwoNearZ * ReciprocalWidth,
            TwoNearZ * ReciprocalHeight,
            -fRange * NearZ,
            0.0,
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // TwoNearZ * ReciprocalWidth,0,0,0
        M.r[0] = vTemp;
        // 0,TwoNearZ * ReciprocalHeight,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // 0,0,fRange,1.0f
        M.r[2] = XMVectorSet(-(ViewLeft + ViewRight) * ReciprocalWidth, 
            -(ViewTop + ViewBottom) * ReciprocalHeight, 
            fRange, 
            1.0);
        // 0,0,-fRange * NearZ,0.0f
        vValues = _mm_and_ps(vValues, *g_XMMaskZ);
        M.r[3] = vValues;
        return M;
    }
}


/// Builds a custom version of a right-handed perspective projection matrix.
/// 
/// ## Parameters
/// 
/// `ViewLeft` The x-coordinate of the left side of the clipping frustum at the near clipping plane.
/// 
/// `ViewRight` The x-coordinate of the right side of the clipping frustum at the near clipping plane.
/// 
/// `ViewBottom` The y-coordinate of the bottom side of the clipping frustum at the near clipping plane.
/// 
/// `ViewTop` The y-coordinate of the top side of the clipping frustum at the near clipping plane.
/// 
/// `NearZ` Distance to the near clipping plane. Must be greater than zero.
/// 
/// `FarZ` Distance to the far clipping plane. Must be greater than zero.
/// 
/// ## Return value
/// 
/// Returns the custom perspective projection matrix.
/// 
/// ## Remarks
/// 
/// For typical usage, NearZ is less than FarZ. However, if you flip these values so FarZ is less than NearZ,
/// the result is an inverted z buffer (also known as a "reverse z buffer") which can provide increased floating-point precision.
///
/// NearZ and FarZ cannot be the same value and must be greater than 0.
/// 
/// ## Reference
/// 
/// <https://learn.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-xmmatrixperspectiveoffcenterrh>
#[inline]
pub fn XMMatrixPerspectiveOffCenterRH(
    ViewLeft: f32,
    ViewRight: f32,
    ViewBottom: f32,
    ViewTop: f32,
    NearZ: f32,
    FarZ: f32,
) -> XMMATRIX {
    debug_assert!(NearZ > 0.0 && FarZ > 0.0);
    debug_assert!(!XMScalarNearEqual(ViewRight, ViewLeft, 0.00001));
    debug_assert!(!XMScalarNearEqual(ViewTop, ViewBottom, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let TwoNearZ = NearZ + NearZ;
        let ReciprocalWidth = 1.0 / (ViewRight - ViewLeft);
        let ReciprocalHeight = 1.0 / (ViewTop - ViewBottom);
        let fRange = FarZ / (NearZ - FarZ);

        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = TwoNearZ * ReciprocalWidth;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = TwoNearZ * ReciprocalHeight;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = (ViewLeft + ViewRight) * ReciprocalWidth;
        M.m[2][1] = (ViewTop + ViewBottom) * ReciprocalHeight;
        M.m[2][2] = fRange;
        M.m[2][3] = -1.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = fRange * NearZ;
        M.m[3][3] = 0.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        let TwoNearZ = NearZ + NearZ;
        let ReciprocalWidth = 1.0 / (ViewRight - ViewLeft);
        let ReciprocalHeight = 1.0 / (ViewTop - ViewBottom);
        let fRange = FarZ / (NearZ - FarZ);
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            TwoNearZ * ReciprocalWidth,
            TwoNearZ * ReciprocalHeight,
            fRange * NearZ,
            0.0,
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // TwoNearZ*ReciprocalWidth,0,0,0
        M.r[0] = vTemp;
        // 0,TwoNearZ*ReciprocalHeight,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // 0,0,fRange,1.0f
        M.r[2] = XMVectorSet((ViewLeft + ViewRight) * ReciprocalWidth,
            (ViewTop + ViewBottom) * ReciprocalHeight,
            fRange,
            -1.0);
        // 0,0,-fRange * NearZ,0.0f
        vValues = _mm_and_ps(vValues, *g_XMMaskZ);
        M.r[3] = vValues;
        return M;
    }
}

/// Builds an orthogonal projection matrix for a left-handed coordinate system.
///
/// ## Parameters
///
/// `ViewWidth` Width of the frustum at the near clipping plane.
///
/// `ViewHeight` Height of the frustum at the near clipping plane.
///
/// `NearZ` Distance to the near clipping plane.
///
/// `FarZ` Distance to the far clipping plane.
///
/// ## Return value
///
/// Returns the orthogonal projection matrix.
///
/// ## Remarks
///
/// All the parameters of XMMatrixOrthographicLH are distances in camera space.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixOrthographicLH>
#[inline]
pub fn XMMatrixOrthographicLH(
    ViewWidth: f32,
    ViewHeight: f32,
    NearZ: f32,
    FarZ: f32
) -> XMMATRIX
{
    debug_assert!(!XMScalarNearEqual(ViewWidth, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(ViewHeight, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let fRange: f32 = 1.0 / (FarZ - NearZ);
        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = 2.0 / ViewWidth;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = 2.0 / ViewHeight;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = fRange;
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = -fRange * NearZ;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        let fRange: f32 = 1.0 / (FarZ - NearZ);
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            2.0 / ViewWidth,
            2.0 / ViewHeight,
            fRange,
            -fRange * NearZ
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // 2.0f / ViewWidth,0,0,0
        M.r[0] = vTemp;
        // 0,2.0f / ViewHeight,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // x=fRange,y=-fRange * NearZ,0,1.0f
        vTemp = _mm_setzero_ps();
        vValues = _mm_shuffle_ps(vValues, *g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
        // 0,0,fRange,0.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 0, 0, 0));
        M.r[2] = vTemp;
        // 0,0,-fRange * NearZ,1.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 1, 0, 0));
        M.r[3] = vTemp;
        return M;
    }
}

/// Builds an orthogonal projection matrix for a right-handed coordinate system.
///
/// ## Parameters
///
/// `ViewWidth` Width of the frustum at the near clipping plane.
///
/// `ViewHeight` Height of the frustum at the near clipping plane.
///
/// `NearZ` Distance to the near clipping plane.
///
/// `FarZ` Distance to the far clipping plane.
///
/// ## Return value
///
/// Returns the orthogonal projection matrix.
///
/// ## Remarks
///
/// All the parameters of XMMatrixOrthographicRH are distances in camera space.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixOrthographicRH>
#[inline]
pub fn XMMatrixOrthographicRH(
    ViewWidth: f32,
    ViewHeight: f32,
    NearZ: f32,
    FarZ: f32
) -> XMMATRIX
{
    debug_assert!(!XMScalarNearEqual(ViewWidth, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(ViewHeight, 0.0, 0.00001));
    debug_assert!(!XMScalarNearEqual(FarZ, NearZ, 0.00001));

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let fRange: f32 = 1.0 / (NearZ - FarZ);
        let mut M: XMMATRIX = crate::undefined();
        M.m[0][0] = 2.0 / ViewWidth;
        M.m[0][1] = 0.0;
        M.m[0][2] = 0.0;
        M.m[0][3] = 0.0;

        M.m[1][0] = 0.0;
        M.m[1][1] = 2.0 / ViewHeight;
        M.m[1][2] = 0.0;
        M.m[1][3] = 0.0;

        M.m[2][0] = 0.0;
        M.m[2][1] = 0.0;
        M.m[2][2] = fRange;
        M.m[2][3] = 0.0;

        M.m[3][0] = 0.0;
        M.m[3][1] = 0.0;
        M.m[3][2] = fRange * NearZ;
        M.m[3][3] = 1.0;
        return M;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut M: XMMATRIX = crate::undefined();
        let fRange: f32 = 1.0 / (NearZ - FarZ);
        // Note: This is recorded on the stack
        let rMem: XMVECTORF32 = XMVECTORF32 { f: [
            2.0 / ViewWidth,
            2.0 / ViewHeight,
            fRange,
            fRange * NearZ
        ]};
        // Copy from memory to SSE register
        let mut vValues: XMVECTOR = rMem.v;
        let mut vTemp: XMVECTOR = _mm_setzero_ps();
        // Copy x only
        vTemp = _mm_move_ss(vTemp, vValues);
        // 2.0f / ViewWidth,0,0,0
        M.r[0] = vTemp;
        // 0,2.0f / ViewHeight,0,0
        vTemp = vValues;
        vTemp = _mm_and_ps(vTemp, *g_XMMaskY);
        M.r[1] = vTemp;
        // x=fRange,y=fRange * NearZ,0,1.0f
        vTemp = _mm_setzero_ps();
        vValues = _mm_shuffle_ps(vValues, *g_XMIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
        // 0,0,fRange,0.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(2, 0, 0, 0));
        M.r[2] = vTemp;
        // 0,0,fRange * NearZ,1.0f
        vTemp = _mm_shuffle_ps(vTemp, vValues, _MM_SHUFFLE(3, 1, 0, 0));
        M.r[3] = vTemp;
        return M;
    }
}

// TODO: XMMatrixOrthographicOffCenterLH
// TODO: XMMatrixOrthographicOffCenterRH

impl std::ops::Deref for XMMatrix {
    type Target = XMMATRIX;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for XMMatrix {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl XMMatrix {
    #[inline]
    pub fn set(
        m00: f32, m01: f32, m02: f32, m03: f32,
        m10: f32, m11: f32, m12: f32, m13: f32,
        m20: f32, m21: f32, m22: f32, m23: f32,
        m30: f32, m31: f32, m32: f32, m33: f32,
    ) -> XMMatrix
    {
        XMMatrix(XMMatrixSet(
            m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33,
        ))
    }
}

impl From<&[f32; 16]> for XMMatrix {
    #[inline]
    fn from(m: &[f32; 16]) -> XMMatrix {
        XMMatrix(XMLoadFloat4x4(m.into()))
    }
}

impl From<&[[f32; 4]; 4]> for XMMatrix {
    #[inline]
    fn from(m: &[[f32; 4]; 4]) -> XMMatrix {
        XMMatrix(XMLoadFloat4x4(m.into()))
    }
}

impl Into<[f32; 16]> for XMMatrix {
    #[inline]
    fn into(self) -> [f32; 16] {
        unsafe {
            let mut R: XMFLOAT4X4 = crate::undefined();
            XMStoreFloat4x4(&mut R, self.0);
            mem::transmute(R)
        }
    }
}

impl Into<[[f32; 4]; 4]> for XMMatrix {
    #[inline]
    fn into(self) -> [[f32; 4]; 4] {
        unsafe {
            let mut R: XMFLOAT4X4 = crate::undefined();
            XMStoreFloat4x4(&mut R, self.0);
            mem::transmute(R)
        }
    }
}

impl std::ops::Add for XMMatrix {
    type Output = XMMatrix;
    #[inline]
    fn add(self, M: XMMatrix) -> Self::Output {
        unsafe {
            XMMatrix(XMMATRIX { r: [
                XMVectorAdd(self.r[0], M.r[0]),
                XMVectorAdd(self.r[1], M.r[1]),
                XMVectorAdd(self.r[2], M.r[2]),
                XMVectorAdd(self.r[3], M.r[3]),
            ]})
        }
    }
}

impl std::ops::AddAssign for XMMatrix {
    #[inline]
    fn add_assign(&mut self, M: XMMatrix) {
        unsafe {
            self.r[0] = XMVectorAdd(self.r[0], M.r[0]);
            self.r[0] = XMVectorAdd(self.r[1], M.r[1]);
            self.r[0] = XMVectorAdd(self.r[2], M.r[2]);
            self.r[0] = XMVectorAdd(self.r[3], M.r[3]);
        }
    }
}

impl std::ops::Sub for XMMatrix {
    type Output = XMMatrix;
    #[inline]
    fn sub(self, M: XMMatrix) -> XMMatrix{
        unsafe {
            XMMatrix(XMMATRIX { r: [
                XMVectorSubtract(self.r[0], M.r[0]),
                XMVectorSubtract(self.r[1], M.r[1]),
                XMVectorSubtract(self.r[2], M.r[2]),
                XMVectorSubtract(self.r[3], M.r[3]),
            ]})
        }
    }
}

impl std::ops::SubAssign for XMMatrix {
    #[inline]
    fn sub_assign(&mut self, M: XMMatrix) {
        unsafe {
            self.r[0] = XMVectorSubtract(self.r[0], M.r[0]);
            self.r[0] = XMVectorSubtract(self.r[1], M.r[1]);
            self.r[0] = XMVectorSubtract(self.r[2], M.r[2]);
            self.r[0] = XMVectorSubtract(self.r[3], M.r[3]);
        }
    }
}

impl std::ops::Mul for XMMatrix {
    type Output = XMMatrix;
    #[inline]
    fn mul(self, M: XMMatrix) -> XMMatrix{
        XMMatrix(XMMatrixMultiply(self.0, &M))
    }
}

impl std::ops::MulAssign for XMMatrix {
    #[inline]
    fn mul_assign(&mut self, M: XMMatrix) {
        self.0 = XMMatrixMultiply(self.0, &M)
    }
}

impl std::ops::Mul<XMMatrix> for f32 {
    type Output = XMMatrix;
    #[inline]
    fn mul(self, M: XMMatrix) -> XMMatrix {
        unsafe {
            let S = self;
            let mut R: XMMATRIX = crate::undefined();
            R.r[0] = XMVectorScale(M.r[0], S);
            R.r[1] = XMVectorScale(M.r[1], S);
            R.r[2] = XMVectorScale(M.r[2], S);
            R.r[3] = XMVectorScale(M.r[3], S);
            return XMMatrix(R);
        }
    }
}

impl std::ops::Mul<f32> for XMMatrix {
    type Output = XMMatrix;
    #[inline]
    fn mul(self, S: f32) -> XMMatrix {
        unsafe {
            let mut R: XMMATRIX = crate::undefined();
            R.r[0] = XMVectorScale(self.r[0], S);
            R.r[1] = XMVectorScale(self.r[1], S);
            R.r[2] = XMVectorScale(self.r[2], S);
            R.r[3] = XMVectorScale(self.r[3], S);
            return XMMatrix(R);
        }
    }
}

impl std::ops::MulAssign<f32> for XMMatrix {
    #[inline]
    fn mul_assign(&mut self, S: f32) {
        unsafe {
            self.r[0] = XMVectorScale(self.r[0], S);
            self.r[1] = XMVectorScale(self.r[1], S);
            self.r[2] = XMVectorScale(self.r[2], S);
            self.r[3] = XMVectorScale(self.r[3], S);
        }
    }
}

impl std::ops::Div<f32> for XMMatrix {
    type Output = XMMatrix;
    #[inline]
    fn div(self, S: f32) -> XMMatrix {
        #[cfg(_XM_NO_INTRINSICS_)]
        unsafe {
            let vS: XMVECTOR = XMVectorReplicate(S);
            let mut R: XMMATRIX = crate::undefined();
            R.r[0] = XMVectorDivide(self.r[0], vS);
            R.r[1] = XMVectorDivide(self.r[1], vS);
            R.r[2] = XMVectorDivide(self.r[2], vS);
            R.r[3] = XMVectorDivide(self.r[3], vS);
            return XMMatrix(R);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_SSE_INTRINSICS_)]
        unsafe {
            let vS: __m128 = _mm_set_ps1(S);
            let mut R: XMMATRIX = crate::undefined();
            R.r[0] = _mm_div_ps(self.r[0], vS);
            R.r[1] = _mm_div_ps(self.r[1], vS);
            R.r[2] = _mm_div_ps(self.r[2], vS);
            R.r[3] = _mm_div_ps(self.r[3], vS);
            return XMMatrix(R);
        }
    }
}

impl std::ops::DivAssign<f32> for XMMatrix {
    #[inline]
    fn div_assign(&mut self, S: f32) {
        #[cfg(_XM_NO_INTRINSICS_)]
        unsafe {
            let vS: XMVECTOR = XMVectorReplicate(S);
            self.r[0] = XMVectorDivide(self.r[0], vS);
            self.r[1] = XMVectorDivide(self.r[1], vS);
            self.r[2] = XMVectorDivide(self.r[2], vS);
            self.r[3] = XMVectorDivide(self.r[3], vS);
        }

        #[cfg(_XM_ARM_NEON_INTRINSICS_)]
        {
            unimplemented!()
        }

        #[cfg(_XM_SSE_INTRINSICS_)]
        unsafe {
            let vS: __m128 = _mm_set_ps1(S);
            self.r[0] = _mm_div_ps(self.r[0], vS);
            self.r[1] = _mm_div_ps(self.r[1], vS);
            self.r[2] = _mm_div_ps(self.r[2], vS);
            self.r[3] = _mm_div_ps(self.r[3], vS);
        }
    }
}

impl std::ops::Neg for XMMatrix {
    type Output = XMMatrix;
    #[inline]
    fn neg(self) -> Self::Output {
        unsafe {
            let mut R: XMMATRIX = crate::undefined();
            R.r[0] = XMVectorNegate(self.r[0]);
            R.r[1] = XMVectorNegate(self.r[1]);
            R.r[2] = XMVectorNegate(self.r[2]);
            R.r[3] = XMVectorNegate(self.r[3]);
            XMMatrix(R)
        }
    }
}

impl std::cmp::PartialEq for XMMatrix {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            XMVector(self.r[0]) == XMVector(rhs.r[0]) &&
            XMVector(self.r[1]) == XMVector(rhs.r[1]) &&
            XMVector(self.r[2]) == XMVector(rhs.r[2]) &&
            XMVector(self.r[3]) == XMVector(rhs.r[3])
        }
    }
}

impl std::fmt::Debug for XMMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let rows = unsafe {
            &[
                [
                    XMVectorGetX(self.r[0]),
                    XMVectorGetY(self.r[0]),
                    XMVectorGetZ(self.r[0]),
                    XMVectorGetW(self.r[0]),
                ],
                [
                    XMVectorGetX(self.r[1]),
                    XMVectorGetY(self.r[1]),
                    XMVectorGetZ(self.r[1]),
                    XMVectorGetW(self.r[1]),
                ],
                [
                    XMVectorGetX(self.r[2]),
                    XMVectorGetY(self.r[2]),
                    XMVectorGetZ(self.r[2]),
                    XMVectorGetW(self.r[2]),
                ],
                [
                    XMVectorGetX(self.r[3]),
                    XMVectorGetY(self.r[3]),
                    XMVectorGetZ(self.r[3]),
                    XMVectorGetW(self.r[3]),
                ],
            ]
        };
        f.debug_list()
            .entries(rows)
            .finish()
    }
}

#[test]
fn test_debug() {
    #[rustfmt::skip]
    let m = XMMatrix::from(&[
        [ 1.0,  2.0,  3.0,  4.0],
        [ 5.0,  6.0,  7.0,  8.0],
        [ 9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
    let s = format!("{:?}", m);
    assert_eq!("[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]", s);
}