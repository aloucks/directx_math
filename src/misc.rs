use crate::*;

/// Tests whether two quaternions are equal.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionEqual>
#[inline]
pub fn XMQuaternionEqual(
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
) -> bool
{
    return XMVector4Equal(Q1, Q2);
}

/// Tests whether two quaternions are not equal.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionNotEqual>
#[inline]
pub fn XMQuaternionNotEqual(
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
) -> bool
{
    return XMVector4NotEqual(Q1, Q2);
}

/// Test whether any component of a quaternion is a NaN.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionIsNaN>
#[inline]
pub fn XMQuaternionIsNaN(
    Q: FXMVECTOR,
) -> bool
{
    return XMVector4IsNaN(Q);
}

/// Test whether any component of a quaternion is either positive or negative infinity.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionIsInfinite>
#[inline]
pub fn XMQuaternionIsInfinite(
    Q: FXMVECTOR,
) -> bool
{
    return XMVector4IsInfinite(Q);
}

/// Tests whether a specific quaternion is the identity quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionIsIdentity>
#[inline]
pub fn XMQuaternionIsIdentity(
    Q: FXMVECTOR,
) -> bool
{
    unsafe {
        return XMVector4Equal(Q, g_XMIdentityR3.v);
    }
}

/// Computes the dot product of two quaternions.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionDot>
#[inline]
pub fn XMQuaternionDot(
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
) -> FXMVECTOR
{
    return XMVector4Dot(Q1, Q2);
}

/// Computes the product of two quaternions.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionMultiply>
#[inline]
pub fn XMQuaternionMultiply(
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let Result = XMVECTORF32 {
            f: [
                (Q2.vector4_f32[3] * Q1.vector4_f32[0]) + (Q2.vector4_f32[0] * Q1.vector4_f32[3]) + (Q2.vector4_f32[1] * Q1.vector4_f32[2]) - (Q2.vector4_f32[2] * Q1.vector4_f32[1]),
                (Q2.vector4_f32[3] * Q1.vector4_f32[1]) - (Q2.vector4_f32[0] * Q1.vector4_f32[2]) + (Q2.vector4_f32[1] * Q1.vector4_f32[3]) + (Q2.vector4_f32[2] * Q1.vector4_f32[0]),
                (Q2.vector4_f32[3] * Q1.vector4_f32[2]) + (Q2.vector4_f32[0] * Q1.vector4_f32[1]) - (Q2.vector4_f32[1] * Q1.vector4_f32[0]) + (Q2.vector4_f32[2] * Q1.vector4_f32[3]),
                (Q2.vector4_f32[3] * Q1.vector4_f32[3]) - (Q2.vector4_f32[0] * Q1.vector4_f32[0]) - (Q2.vector4_f32[1] * Q1.vector4_f32[1]) - (Q2.vector4_f32[2] * Q1.vector4_f32[2])
            ]
        };
        return Result.v;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // TODO: (PERFORMANCE) These are defined as static const. Does it matter?
        const ControlWZYX: XMVECTORF32 = XMVECTORF32 { f: [ 1.0, -1.0, 1.0, -1.0 ] };
        const ControlZWXY: XMVECTORF32 = XMVECTORF32 { f: [ 1.0, 1.0, -1.0, -1.0 ] };
        const ControlYXWZ: XMVECTORF32 = XMVECTORF32 { f: [ -1.0, 1.0, 1.0, -1.0 ] };
        // Copy to SSE registers and use as few as possible for x86
        let mut Q2X: XMVECTOR = Q2;
        let mut Q2Y: XMVECTOR = Q2;
        let mut Q2Z: XMVECTOR = Q2;
        let mut vResult: XMVECTOR = Q2;
        // Splat with one instruction
        vResult = XM_PERMUTE_PS!(vResult, _MM_SHUFFLE(3, 3, 3, 3));
        Q2X = XM_PERMUTE_PS!(Q2X, _MM_SHUFFLE(0, 0, 0, 0));
        Q2Y = XM_PERMUTE_PS!(Q2Y, _MM_SHUFFLE(1, 1, 1, 1));
        Q2Z = XM_PERMUTE_PS!(Q2Z, _MM_SHUFFLE(2, 2, 2, 2));
        // Retire Q1 and perform Q1*Q2W
        vResult = _mm_mul_ps(vResult, Q1);
        let mut Q1Shuffle: XMVECTOR = Q1;
        // Shuffle the copies of Q1
        Q1Shuffle = XM_PERMUTE_PS!(Q1Shuffle, _MM_SHUFFLE(0, 1, 2, 3));
        // Mul by Q1WZYX
        Q2X = _mm_mul_ps(Q2X, Q1Shuffle);
        Q1Shuffle = XM_PERMUTE_PS!(Q1Shuffle, _MM_SHUFFLE(2, 3, 0, 1));
        // Flip the signs on y and z
        vResult = XM_FMADD_PS!(Q2X, ControlWZYX.v, vResult);
        // Mul by Q1ZWXY
        Q2Y = _mm_mul_ps(Q2Y, Q1Shuffle);
        Q1Shuffle = XM_PERMUTE_PS!(Q1Shuffle, _MM_SHUFFLE(0, 1, 2, 3));
        // Flip the signs on z and w
        Q2Y = _mm_mul_ps(Q2Y, ControlZWXY.v);
        // Mul by Q1YXWZ
        Q2Z = _mm_mul_ps(Q2Z, Q1Shuffle);
        // Flip the signs on x and w
        Q2Y = XM_FMADD_PS!(Q2Z, ControlYXWZ.v, Q2Y);
        vResult = _mm_add_ps(vResult, Q2Y);
        return vResult;
    }
}

/// Computes the square of the magnitude of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionLengthSq>
#[inline]
pub fn XMQuaternionLengthSq(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    return XMVector4LengthSq(Q);
}

/// Computes the reciprocal of the magnitude of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionReciprocalLength>
#[inline]
pub fn XMQuaternionReciprocalLength(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    return XMVector4ReciprocalLength(Q);
}

/// Computes the magnitude of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionLength>
#[inline]
pub fn XMQuaternionLength(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    return XMVector4Length(Q);
}

/// Estimates the normalized version of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionNormalizeEst>
#[inline]
pub fn XMQuaternionNormalizeEst(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    return XMVector4NormalizeEst(Q);
}

/// Computes the normalized version of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionNormalize>
#[inline]
pub fn XMQuaternionNormalize(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    return XMVector4Normalize(Q);
}

/// Computes the conjugate of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionConjugate>
#[inline]
pub fn XMQuaternionConjugate(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let Result = XMVECTORF32 {
            f: [
                -Q.vector4_f32[0],
                -Q.vector4_f32[1],
                -Q.vector4_f32[2],
                Q.vector4_f32[3]
            ]
        };
        return Result.v;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // TODO: (PERFORMANCE) This is defined as static const
        const NegativeOne3: XMVECTORF32 = XMVECTORF32 { f: [-1.0, -1.0, -1.0, 1.0 ] };
        return _mm_mul_ps(Q, NegativeOne3.v)
    }
}

/// Computes the inverse of a quaternion.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionInverse>
#[inline]
pub fn XMQuaternionInverse(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    unsafe {
        // const XMVECTOR  Zero = XMVectorZero();

        let L: XMVECTOR = XMVector4LengthSq(Q);
        let Conjugate: XMVECTOR = XMQuaternionConjugate(Q);

        let Control: XMVECTOR = XMVectorLessOrEqual(L, g_XMEpsilon.v);

        let mut Result: XMVECTOR = XMVectorDivide(Conjugate, L);

        Result = XMVectorSelect(Result, g_XMZero.v, Control);

        return Result;
    }
}

// TODO: XMQuaternionLn
// TODO: XMQuaternionExp


/// Interpolates between two unit quaternions, using spherical linear interpolation.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionSlerp>
#[inline]
pub fn XMQuaternionSlerp(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    t: f32,
) -> FXMVECTOR
{
    let T: XMVECTOR = XMVectorReplicate(t);
    return XMQuaternionSlerpV(Q0, Q1, T);
}

/// Interpolates between two unit quaternions, using spherical linear interpolation.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionSlerpV>
#[inline]
pub fn XMQuaternionSlerpV(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    T: FXMVECTOR,
) -> FXMVECTOR
{
    // debug_assert!((XMVectorGetY(T) == XMVectorGetX(T)) && (XMVectorGetZ(T) == XMVectorGetX(T)) && (XMVectorGetW(T) == XMVectorGetX(T))));

    #[cfg(any(_XM_NO_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        // TODO: PERFORMANCE These are defined as static const
        const OneMinusEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ 1.0 - 0.00001, 1.0 - 0.00001, 1.0 - 0.00001, 1.0 - 0.00001 ] };

        let mut CosOmega: XMVECTOR = XMQuaternionDot(Q0, Q1);

        // const let mut Zero: XMVECTOR = XMVectorZero();
        const Zero: XMVECTOR = unsafe { g_XMZero.v };
        let mut Control: XMVECTOR = XMVectorLess(CosOmega, Zero);
        let Sign: XMVECTOR = XMVectorSelect(g_XMOne.v, g_XMNegativeOne.v, Control);

        CosOmega = XMVectorMultiply(CosOmega, Sign);

        Control = XMVectorLess(CosOmega, OneMinusEpsilon.v);

        let mut SinOmega: XMVECTOR = XMVectorNegativeMultiplySubtract(CosOmega, CosOmega, g_XMOne.v);
        SinOmega = XMVectorSqrt(SinOmega);

        let Omega: XMVECTOR = XMVectorATan2(SinOmega, CosOmega);

        let mut SignMask: XMVECTOR = XMVectorSplatSignMask();
        let mut V01: XMVECTOR = XMVectorShiftLeft(T, Zero, 2);
        SignMask = XMVectorShiftLeft(SignMask, Zero, 3);
        V01 = XMVectorXorInt(V01, SignMask);
        V01 = XMVectorAdd(g_XMIdentityR0.v, V01);

        let InvSinOmega: XMVECTOR = XMVectorReciprocal(SinOmega);

        let mut S0: XMVECTOR = XMVectorMultiply(V01, Omega);
        S0 = XMVectorSin(S0);
        S0 = XMVectorMultiply(S0, InvSinOmega);

        S0 = XMVectorSelect(V01, S0, Control);

        let mut S1: XMVECTOR = XMVectorSplatY(S0);
        S0 = XMVectorSplatX(S0);

        S1 = XMVectorMultiply(S1, Sign);

        let mut Result: XMVECTOR = XMVectorMultiply(Q0, S0);
        Result = XMVectorMultiplyAdd(Q1, S1, Result);

        return Result;
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // TODO: PERFORMANCE These are defined as static const
        const OneMinusEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ 1.0 - 0.00001, 1.0 - 0.00001, 1.0 - 0.00001, 1.0 - 0.00001 ] };
        const SignMask2: XMVECTORU32 = XMVECTORU32 { u: [ 0x80000000, 0x00000000, 0x00000000, 0x00000000 ] };

        let mut CosOmega: XMVECTOR = XMQuaternionDot(Q0, Q1);

        // const let mut Zero: XMVECTOR = XMVectorZero();
        const Zero: XMVECTOR = unsafe { g_XMZero.v };

        let mut Control: XMVECTOR = XMVectorLess(CosOmega, Zero);
        let Sign: XMVECTOR = XMVectorSelect(g_XMOne.v, g_XMNegativeOne.v, Control);

        CosOmega = _mm_mul_ps(CosOmega, Sign);

        Control = XMVectorLess(CosOmega, OneMinusEpsilon.v);

        let mut SinOmega: XMVECTOR = _mm_mul_ps(CosOmega, CosOmega);
        SinOmega = _mm_sub_ps(g_XMOne.v, SinOmega);
        SinOmega = _mm_sqrt_ps(SinOmega);

        let Omega: XMVECTOR = XMVectorATan2(SinOmega, CosOmega);

        let mut V01: XMVECTOR = XM_PERMUTE_PS!(T, _MM_SHUFFLE(2, 3, 0, 1));
        V01 = _mm_and_ps(V01, g_XMMaskXY.v);
        V01 = _mm_xor_ps(V01, SignMask2.v);
        V01 = _mm_add_ps(g_XMIdentityR0.v, V01);

        let mut S0: XMVECTOR = _mm_mul_ps(V01, Omega);
        S0 = XMVectorSin(S0);
        S0 = _mm_div_ps(S0, SinOmega);

        S0 = XMVectorSelect(V01, S0, Control);

        let mut S1: XMVECTOR = XMVectorSplatY(S0);
        S0 = XMVectorSplatX(S0);

        S1 = _mm_mul_ps(S1, Sign);
        let mut Result: XMVECTOR = _mm_mul_ps(Q0, S0);
        S1 = _mm_mul_ps(S1, Q1);
        Result = _mm_add_ps(Result, S1);
        return Result;
    }
}