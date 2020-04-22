use crate::*;
use std::mem;

/// Tests whether any of the elements of a matrix are NaN.
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
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixMultiply>
#[inline]
pub fn XMMatrixMultiply(
    M1: FXMMATRIX,
    M2: CXMMATRIX,
) -> FXMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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

        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
        mResult.r[0] = _mm256_castps256_ps128(t0);
        mResult.r[1] = _mm256_extractf128_ps(t0, 1);
        mResult.r[2] = _mm256_castps256_ps128(t1);
        mResult.r[3] = _mm256_extractf128_ps(t1, 1);
        return mResult;
    }

    #[cfg(all(_XM_AVX_INTRINSICS_, not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();

        // Splat the component X,Y,Z then W
        let mut vX: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(0));
        let mut vY: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(1));
        let mut vZ: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(2));
        let mut vW: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(3));

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
        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(0));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(1));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(2));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[1] = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(0));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(1));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(2));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        mResult.r[2] = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(0));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(1));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(2));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(3));

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
        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();

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
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixMultiplyTranspose>
#[inline]
pub fn XMMatrixMultiplyTranspose(
    M1: FXMMATRIX,
    M2: CXMMATRIX,
) -> FXMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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

        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
        mResult.r[0] = _mm256_castps256_ps128(t0);
        mResult.r[1] = _mm256_extractf128_ps(t0, 1);
        mResult.r[2] = _mm256_castps256_ps128(t1);
        mResult.r[3] = _mm256_extractf128_ps(t1, 1);
        return mResult;
    }

    #[cfg(all(_XM_AVX_INTRINSICS_, not(_XM_AVX2_INTRINSICS_)))]
    unsafe {
        // Splat the component X,Y,Z then W
        let mut vX: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(0));
        let mut vY: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(1));
        let mut vZ: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(2));
        let mut vW: XMVECTOR = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[0]).add(3));

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
        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(0));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(1));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(2));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[1]).add(3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r1: XMVECTOR = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(0));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(1));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(2));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[2]).add(3));

        vX = _mm_mul_ps(vX, M2.r[0]);
        vY = _mm_mul_ps(vY, M2.r[1]);
        vZ = _mm_mul_ps(vZ, M2.r[2]);
        vW = _mm_mul_ps(vW, M2.r[3]);

        vX = _mm_add_ps(vX, vZ);
        vY = _mm_add_ps(vY, vW);
        vX = _mm_add_ps(vX, vY);
        let r2: XMVECTOR = vX;

        vX = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(0));
        vY = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(1));
        vZ = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(2));
        vW = _mm_broadcast_ss(&*mem::transmute::<_, *const f32>(&M1.r[3]).add(3));

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

        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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

        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMMatrixTranspose>
#[inline]
pub fn XMMatrixTranspose(
    M: FXMMATRIX,
) -> FXMMATRIX
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        // Original matrix:
        //
        //     m00m01m02m03
        //     m10m11m12m13
        //     m20m21m22m23
        //     m30m31m32m33

        let mut P: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
        P.r[0] = XMVectorMergeXY(M.r[0], M.r[2]); // m00m20m01m21
        P.r[1] = XMVectorMergeXY(M.r[1], M.r[3]); // m10m30m11m31
        P.r[2] = XMVectorMergeZW(M.r[0], M.r[2]); // m02m22m03m23
        P.r[3] = XMVectorMergeZW(M.r[1], M.r[3]); // m12m32m13m33

        let mut MT: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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

        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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

        let mut mResult: XMMATRIX = mem::MaybeUninit::uninit().assume_init();
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