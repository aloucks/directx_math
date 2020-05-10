use crate::*;

/// Tests whether two quaternions are equal.
///
/// ## Parameters
///
/// `Q1` First quaternion to test.
///
/// `Q2` Second quaternion to test.
///
/// ## Return value
///
/// Returns `true` if the quaternions are equal and `false` otherwise.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q1` First quaternion to test.
///
/// `Q2` Second quaternion to test.
///
/// ## Return value
///
/// Returns `true` if the quaternions are unequal and `false` otherwise.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` Quaternion to test.
///
/// ## Return value
///
/// Returns `true` if any component of `Q` is a `NaN`, and `false` otherwise.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` Quaternion to test.
///
/// ## Return value
///
/// Returns `true` if any component of `Q` is positive or negative infinity, and `false` otherwise.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` Quaternion to test.
///
/// ## Return value
///
/// Returns `true` if `Q` is the identity quaternion, or `false` otherwise.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q1` First quaternion
///
/// `Q2` Second quaternion.
///
/// ## Return value
///
/// Returns a vector. The dot product between `Q1` and `Q2` is replicated into each component.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q1` First quaternion.
///
/// `Q2` Second quaternion.
///
/// ## Return Value
///
/// Returns the product of two quaternions as `Q2*Q1`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent
/// quaternions, where the `X`, `Y`, and `Z` components are the vector part and
/// the `W` component is the scalar part.
///
/// The result represents the rotation `Q1` followed by the rotation `Q2` to be
/// consistent with [`XMMatrixMultiply`] concatenation since this function is
/// typically used to concatenate quaternions that represent rotations
/// (i.e. it returns `Q2*Q1`).
///
/// This function computes the equivalent to the following pseduo-code:
///
/// ```text
/// XMVECTOR Result;
/// Result.x = (Q2.w * Q1.x) + (Q2.x * Q1.w) + (Q2.y * Q1.z) - (Q2.z * Q1.y);
/// Result.y = (Q2.w * Q1.y) - (Q2.x * Q1.z) + (Q2.y * Q1.w) + (Q2.z * Q1.x);
/// Result.z = (Q2.w * Q1.z) + (Q2.x * Q1.y) - (Q2.y * Q1.x) + (Q2.z * Q1.w);
/// Result.w = (Q2.w * Q1.w) - (Q2.x * Q1.x) - (Q2.y * Q1.y) - (Q2.z * Q1.z);
/// return Result;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionMultiply>
///
/// [`XMMatrixMultiply`]: crate::matrix::XMMatrixMultiply
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
/// ## Parameters
///
/// `Q` Quaternion to measure.
///
/// ## Return value
///
/// Returns a vector. The square of the magnitude of `Q` is replicated into each component.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` Quaternion to measure.
///
/// ## Return value
///
/// Returns the reciprocal of the magnitude of `Q`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` Quaternion to measure.
///
/// ## Return value
///
/// Returns a vector. The magnitude of `Q` is replicated into each component.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` A quaternion for which to estimate the normalized version.
///
/// ## Return value
///
/// An XMVECTOR union that is the estimate of the normalized version of a quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// This function internally calls the XMVector4NormalizeEstfunction.
///
/// `Est` functions offer increased performance at the expense of reduced accuracy. `Est` functions are appropriate
/// for non-critical calculations where accuracy can be sacrificed for speed. The exact amount of lost accuracy
/// and speed increase are platform dependent.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` Quaternion to normalize.
///
/// ## Return value
///
/// Returns the normalized form of `Q`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q` The quaternion to conjugate.
///
/// ## Return value
///
/// Returns the conjugate of `Q`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// Given a quaternion (`x`, `y`, `z`, `w`), the [`XMQuaternionConjugate`] function returns the
/// quaternion (`-x`, `-y`, `-z`, `w`).
///
/// Use the [`XMQuaternionNormalize`] function for any quaternion input that is not already normalized.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionConjugate>
///
/// [`XMQuaternionConjugate`]: crate::quaternion::XMQuaternionConjugate
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
/// ## Parameters
///
/// `Q` Quaternion to invert.
///
/// ## Return value
///
/// Returns the inverse of `Q`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// The following pseudocode demonstrates the operation of the function:
///
/// ```text
/// XMVECTOR Result;
///
/// float LengthSq = Q.x * Q.x + Q.y * Q.y + Q.z * Q.z + Q.w * Q.w;
///
/// Result.x = -Q.x / LengthSq;
/// Result.y = -Q.y / LengthSq;
/// Result.z = -Q.z / LengthSq;
/// Result.w = Q.w / LengthSq;
///
/// return Result;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionInverse>
#[inline]
pub fn XMQuaternionInverse(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    unsafe {
        // const Zero: XMVECTOR = XMVectorZero();
        const Zero: XMVECTOR = unsafe { g_XMZero.v };

        let L: XMVECTOR = XMVector4LengthSq(Q);
        let Conjugate: XMVECTOR = XMQuaternionConjugate(Q);

        let Control: XMVECTOR = XMVectorLessOrEqual(L, g_XMEpsilon.v);

        let mut Result: XMVECTOR = XMVectorDivide(Conjugate, L);

        Result = XMVectorSelect(Result, Zero, Control);

        return Result;
    }
}

/// Computes the natural logarithm of a given unit quaternion.
///
/// ## Parameters
///
/// `Q` Unit quaternion for which to calculate the natural logarithm. If `Q` is not a unit quaternion, the returned
/// value is undefined.
///
/// ## Return value
///
/// Returns the natural logarithm of `Q`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionLn>
#[inline]
pub fn XMQuaternionLn(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    unsafe {
        // TODO: PERFORMANCE static const
        const OneMinusEpsilon: XMVECTOR = unsafe { XMVECTORF32 { f: [1.0 - 0.00001, 1.0 - 0.00001, 1.0 - 0.00001, 1.0 - 0.00001] }.v };

        let QW: XMVECTOR = XMVectorSplatW(Q);
        let Q0: XMVECTOR = XMVectorSelect(g_XMSelect1110.v, Q, g_XMSelect1110.v);

        let ControlW: XMVECTOR = XMVectorInBounds(QW, OneMinusEpsilon);

        let Theta: XMVECTOR = XMVectorACos(QW);
        let SinTheta: XMVECTOR = XMVectorSin(Theta);

        let S: XMVECTOR = XMVectorDivide(Theta, SinTheta);

        let mut Result: XMVECTOR = XMVectorMultiply(Q0, S);
        Result = XMVectorSelect(Q0, Result, ControlW);

        return Result;
    }
}

/// Computes the exponential of a given pure quaternion.
///
/// ## Parameters
///
/// `Q` Pure quaternion for which to compute the exponential. The `w`-component of the input
/// quaternion is ignored in the calculation.
///
/// ## Return value
///
/// Returns the exponential of `Q`.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionExp>
#[inline]
pub fn XMQuaternionExp(
    Q: FXMVECTOR,
) -> FXMVECTOR
{
    unsafe {
        let Theta: XMVECTOR = XMVector3Length(Q);

        let mut SinTheta: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        let mut CosTheta: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        XMVectorSinCos(&mut SinTheta, &mut CosTheta, Theta);

        let S: XMVECTOR = XMVectorDivide(SinTheta, Theta);

        let mut Result: XMVECTOR = XMVectorMultiply(Q, S);

        //const let Zero: XMVECTOR = XMVectorZero();
        const Zero: XMVECTOR = unsafe { g_XMZero.v };
        let Control: XMVECTOR = XMVectorNearEqual(Theta, Zero, g_XMEpsilon.v);
        Result = XMVectorSelect(Result, Q, Control);

        Result = XMVectorSelect(CosTheta, Result, g_XMSelect1110.v);

        return Result;
    }
}


/// Interpolates between two unit quaternions, using spherical linear interpolation.
///
/// ## Parameters
///
/// `Q0` Unit quaternion to interpolate from.
///
/// `Q` Unit quaternion to interpolate to.
///
/// `t` Interpolation control factor.
///
/// ## Return value
///
/// Returns the interpolated quaternion. If `Q0` and `Q1` are not unit quaternions, the resulting
/// interpolation is undefined.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// When `t` is `0.0`, the function returns `Q0`. When `t` is `1.0`, the function returns `Q1`.
///
/// ## Reference
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
/// ## Parameters
///
/// `Q0` Unit quaternion to interpolate from.
///
/// `Q1` Unit quaternion to interpolate to.
///
/// `T` Interpolation control factor. All components of this vector must be the same.
///
/// ## Return value
///
/// Returns the interpolated quaternion. If `Q0` and `Q1` are not unit quaternions, the resulting interpolation
/// is undefined.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// This function is identical to [`XMQuaternionSlerp`] except that `T` is supplied using a 4D vector instead
/// of a **float** value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionSlerpV>
///
/// [`XMQuaternionSlerp`]: crate::quaternion::XMQuaternionSlerp
#[inline]
pub fn XMQuaternionSlerpV(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    T: FXMVECTOR,
) -> FXMVECTOR
{
    debug_assert!((XMVectorGetY(T) == XMVectorGetX(T)) && (XMVectorGetZ(T) == XMVectorGetX(T)) && (XMVectorGetW(T) == XMVectorGetX(T)));

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

/// Interpolates between four unit quaternions, using spherical quadrangle interpolation.
///
/// ## Parameters
///
/// `Q0` First unit quaternion.
///
/// `Q1` Second unit quaternion.
///
/// `Q2` Third unit quaternion.
///
/// `Q3` Fourth unit quaternion.
///
/// `t` Interpolation control factor.
///
/// ## Return value
///
/// Returns the interpolated quaternion. If `Q0`, `Q1`, `Q2`, and `Q3` are not
/// all unit quaternions, the returned quaternion is undefined.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent
/// quaternions, where the `X`, `Y`, and `Z` components are the vector part and the
/// `W` component is the scalar part.
///
/// The use of this method requires some setup before its use. See [`XMQuaternionSquadSetup`] for details.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionSquad>
///
/// [`XMQuaternionSquadSetup`]: crate::quaternion::XMQuaternionSquadSetup
#[inline]
pub fn XMQuaternionSquad(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
    Q3: GXMVECTOR,
    t: f32,
) -> FXMVECTOR
{
    let T: XMVECTOR = XMVectorReplicate(t);
    return XMQuaternionSquadV(Q0, Q1, Q2, Q3, T);
}

/// Interpolates between four unit quaternions, using spherical quadrangle interpolation.
///
/// ## Parameters
///
/// `Q0` First unit quaternion.
///
/// `Q1` Second unit quaternion.
///
/// `Q2` Third unit quaternion.
///
/// `Q3` Fourth unit quaternion.
///
/// `T` Interpolation control factor. All components of this vector must be the same.
///
/// ## Return value
///
/// Returns the interpolated quaternion. If Q0, Q1, Q2, and Q3 are not unit quaternions, the resulting interpolation
/// is undefined.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// This function is identical to [`XMQuaternionSquad`] except that `T` is supplied using a 4D vector instead
/// of a **float** value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionSquadV>
///
/// [`XMQuaternionSquad`]: crate::quaternion::XMQuaternionSquad
#[inline]
pub fn XMQuaternionSquadV(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
    Q3: GXMVECTOR,
    T: XMVECTOR,
) -> FXMVECTOR
{
    debug_assert!((XMVectorGetY(T) == XMVectorGetX(T)) && (XMVectorGetZ(T) == XMVectorGetX(T)) && (XMVectorGetW(T) == XMVectorGetX(T)));

    let mut TP: XMVECTOR = T;

    let Two: XMVECTOR = XMVectorSplatConstant(2, 0);

    let Q03: XMVECTOR = XMQuaternionSlerpV(Q0, Q3, T);
    let Q12: XMVECTOR = XMQuaternionSlerpV(Q1, Q2, T);

    TP = XMVectorNegativeMultiplySubtract(TP, TP, TP);
    TP = XMVectorMultiply(TP, Two);

    let Result: XMVECTOR = XMQuaternionSlerpV(Q03, Q12, TP);

    return Result;
}

/// Provides addresses of setup control points for spherical quadrangle interpolation.
///
/// ## Parameters
///
/// `pA` Address of first setup quaternion.
///
/// `pB` Address of first setup quaternion.
///
/// `pC` Address of first setup quaternion.
///
/// `Q0` First quaternion.
///
/// `Q1` Second quaternion.
///
/// `Q2` Third quaternion.
///
/// `Q3` Fourth quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent
/// quaternions, where the `X`, `Y`, and `Z` components are the vector part and the
/// `W` component is the scalar part.
///
/// The results returned in `pA`, `pB`, and `pC` are used as the inputs to the
/// `Q1`, `Q2`, and `Q3` parameters of [`XMQuaternionSquad`].
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionSquadSetup>
///
/// [`XMQuaternionSquad`]: crate::quaternion::XMQuaternionSquad
#[inline]
pub fn XMQuaternionSquadSetup(
    pA: &mut XMVECTOR,
    pB: &mut XMVECTOR,
    pC: &mut XMVECTOR,
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
    Q3: GXMVECTOR
)
{
    let LS12: XMVECTOR = XMQuaternionLengthSq(XMVectorAdd(Q1, Q2));
    let LD12: XMVECTOR = XMQuaternionLengthSq(XMVectorSubtract(Q1, Q2));
    let mut SQ2: XMVECTOR = XMVectorNegate(Q2);

    let Control1: XMVECTOR = XMVectorLess(LS12, LD12);
    SQ2 = XMVectorSelect(Q2, SQ2, Control1);

    let LS01: XMVECTOR = XMQuaternionLengthSq(XMVectorAdd(Q0, Q1));
    let LD01: XMVECTOR = XMQuaternionLengthSq(XMVectorSubtract(Q0, Q1));
    let mut SQ0: XMVECTOR = XMVectorNegate(Q0);

    let LS23: XMVECTOR = XMQuaternionLengthSq(XMVectorAdd(SQ2, Q3));
    let LD23: XMVECTOR = XMQuaternionLengthSq(XMVectorSubtract(SQ2, Q3));
    let mut SQ3: XMVECTOR = XMVectorNegate(Q3);

    let Control0: XMVECTOR = XMVectorLess(LS01, LD01);
    let Control2: XMVECTOR = XMVectorLess(LS23, LD23);

    SQ0 = XMVectorSelect(Q0, SQ0, Control0);
    SQ3 = XMVectorSelect(Q3, SQ3, Control2);

    let InvQ1: XMVECTOR = XMQuaternionInverse(Q1);
    let InvQ2: XMVECTOR = XMQuaternionInverse(SQ2);

    let LnQ0: XMVECTOR = XMQuaternionLn(XMQuaternionMultiply(InvQ1, SQ0));
    let LnQ2: XMVECTOR = XMQuaternionLn(XMQuaternionMultiply(InvQ1, SQ2));
    let LnQ1: XMVECTOR = XMQuaternionLn(XMQuaternionMultiply(InvQ2, Q1));
    let LnQ3: XMVECTOR = XMQuaternionLn(XMQuaternionMultiply(InvQ2, SQ3));

    let NegativeOneQuarter: XMVECTOR = XMVectorSplatConstant(-1, 2);

    let mut ExpQ02: XMVECTOR = XMVectorMultiply(XMVectorAdd(LnQ0, LnQ2), NegativeOneQuarter);
    let mut ExpQ13: XMVECTOR = XMVectorMultiply(XMVectorAdd(LnQ1, LnQ3), NegativeOneQuarter);
    ExpQ02 = XMQuaternionExp(ExpQ02);
    ExpQ13 = XMQuaternionExp(ExpQ13);

    *pA = XMQuaternionMultiply(Q1, ExpQ02);
    *pB = XMQuaternionMultiply(SQ2, ExpQ13);
    *pC = SQ2;
}

/// Returns a point in barycentric coordinates, using the specified quaternions.
///
/// ## Parameters
///
/// `Q0` First quaternion in the triangle.
///
/// `Q1` Second quaternion in the triangle.
///
/// `Q2` Third quaternion in the triangle.
///
/// `f` Weighting factor. See the remarks.
///
/// `g` Weighting factor. See the remarks.
///
/// ## Return value
///
/// Returns a quaternion in barycentric coordinates.
///
/// ## Remarks
///
/// The following pseudocode demonstrates the operation of the function.
///
/// ```text
/// XMVECTOR Result;
/// XMVECTOR QA, QB;
/// float s = f + g;
///
/// if (s != 0.0f)
/// {
///     QA = XMQuaternionSlerp(Q0, Q1, s);
///     QB = XMQuaternionSlerp(Q0, Q2, s);
///     Result = XMQuaternionSlerp(QA, QB, g / s);
/// }
/// else
/// {
///     Result.x = Q0.x;
///     Result.y = Q0.y;
///     Result.z = Q0.z;
///     Result.w = Q0.w;
/// }
///
/// return Result;
/// ```
///
/// Note that Barycentric coordinates work for 'flat' surfaces but not for 'curved' ones.
/// This function is therefore a bit of a work-around. An alternative method for blending
/// 3 quanterions is given by the following code:
///
/// ```text
/// inline XMVECTOR XMQuaternionBlend(
///   FXMVECTOR Q0,
///   FXMVECTOR Q1,
///   FXMVECTOR Q2,
///   float w1,
///   float w2
/// )
/// {
///     // Note if you choose one of the three weights to be zero, you get a blend of two
///     // quaternions.  This does not give you slerp of those quaternions.
///     float w0 = 1.0f - w1 - w2;
///     XMVECTOR Result = XMVector4Normalize(
///         XMVectorScale(Q0, w0) +
///         XMVectorScale(Q1, w1) +
///         XMVectorScale(Q2, w2));
///     return Result;
/// }
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionBaryCentric>
#[inline]
pub fn XMQuaternionBaryCentric(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
    f: f32,
    g: f32,
) -> XMVECTOR
{
    let s = f + g;

    let Result: XMVECTOR;
    if ((s < 0.00001) && (s > -0.00001))
    {
        Result = Q0;
    }
    else
    {
        let Q01: XMVECTOR = XMQuaternionSlerp(Q0, Q1, s);
        let Q02: XMVECTOR = XMQuaternionSlerp(Q0, Q2, s);

        Result = XMQuaternionSlerp(Q01, Q02, g / s);
    }

    return Result;
}

/// Returns a point in barycentric coordinates, using the specified quaternions.
///
/// ## Parameters
///
/// `Q0` First quaternion in the triangle.
///
/// `Q1` Second quaternion in the triangle.
///
/// `Q2` Third quaternion in the triangle.
///
/// `F` Weighting factor. All components of this vector must be the same.
///
/// `G` Weighting factor. All components of this vector must be the same.
///
/// ## Return value
///
/// Returns a quaternion in barycentric coordinates.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// This function is identical to [`XMQuaternionBaryCentric`] except that `F` and `G` are supplied using
/// a 4D vector instead of a **float** value.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionBaryCentricV>
///
/// [`XMQuaternionBaryCentric`]: crate::quaternion::XMQuaternionBaryCentric
#[inline]
pub fn XMQuaternionBaryCentricV(
    Q0: FXMVECTOR,
    Q1: FXMVECTOR,
    Q2: FXMVECTOR,
    F: FXMVECTOR,
    G: FXMVECTOR,
) -> XMVECTOR
{
    debug_assert!((XMVectorGetY(F) == XMVectorGetX(F)) && (XMVectorGetZ(F) == XMVectorGetX(F)) && (XMVectorGetW(F) == XMVectorGetX(F)));
    debug_assert!((XMVectorGetY(G) == XMVectorGetX(G)) && (XMVectorGetZ(G) == XMVectorGetX(G)) && (XMVectorGetW(G) == XMVectorGetX(G)));

    // PERFORMANCE: const
    let Epsilon: XMVECTOR = XMVectorSplatConstant(1, 16);

    let S: XMVECTOR = XMVectorAdd(F, G);

    let Result: XMVECTOR;
    if (XMVector4InBounds(S, Epsilon))
    {
        Result = Q0;
    }
    else
    {
        let Q01: XMVECTOR = XMQuaternionSlerpV(Q0, Q1, S);
        let Q02: XMVECTOR = XMQuaternionSlerpV(Q0, Q2, S);
        let mut GS: XMVECTOR = XMVectorReciprocal(S);
        GS = XMVectorMultiply(G, GS);

        Result = XMQuaternionSlerpV(Q01, Q02, GS);
    }

    return Result;
}

/// Returns the identity quaternion.
///
/// ## Return value
///
/// An XMVECTOR that is the identity quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// Given a quaternion (`x`, `y`, `z`, `w`), the [`XMQuaternionIdentity`] function will return the
/// quaternion (`0`, `0`, `0`, `1`).
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionIdentity>
///
/// [`XMQuaternionIdentity`]: crate::quaternion::XMQuaternionIdentity
#[inline]
pub fn XMQuaternionIdentity() -> FXMVECTOR
{
    unsafe {
        return g_XMIdentityR3.v;
    }
}


/// Computes a rotation quaternion based on the `pitch`, `yaw`, and `roll` (Euler angles).
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
/// Returns the rotation quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// Angles are measured clockwise when looking along the rotation axis toward the origin. **This is a left-handed
/// coordinate system. To use right-handed coordinates, negate all three angles.**
///
/// The order of transformations is `roll` first, then `pitch`, then `yaw`. The rotations are all applied in the
/// global coordinate frame.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionRotationRollPitchYaw>
#[inline]
pub fn XMQuaternionRotationRollPitchYaw(
    Pitch: f32,
    Yaw: f32,
    Roll: f32,
) -> FXMVECTOR
{
    let Angles: XMVECTOR = XMVectorSet(Pitch, Yaw, Roll, 0.0);
    let Q: XMVECTOR = XMQuaternionRotationRollPitchYawFromVector(Angles);
    return Q;
}

/// Computes a rotation quaternion based on a vector containing the Euler angles (`pitch`, `yaw`, and `roll`).
///
/// ## Parameters
///
/// `Angles` 3D vector containing the Euler angles in the order `pitch`, `yaw`, `roll`.
///
/// ## Return value
///
/// Returns the rotation quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// Angles are measured clockwise when looking along the rotation axis toward the origin. **This is a left-handed
/// coordinate system. To use right-handed coordinates, negate all three angles.**
///
/// The order of transformations is `roll` first, then `pitch`, then `yaw`. The rotations are all applied in the
/// global coordinate frame.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionRotationRollPitchYawFromVector>
#[inline]
pub fn XMQuaternionRotationRollPitchYawFromVector(
    Angles: XMVECTOR, // <Pitch, Yaw, Roll, 0>
) -> FXMVECTOR
{
    unsafe {
        // TODO: This is defined as a static const
        const Sign: XMVECTORF32 = XMVECTORF32 { f: [1.0, -1.0, -1.0, 1.0 ] };

        let HalfAngles: XMVECTOR = XMVectorMultiply(Angles, g_XMOneHalf.v);

        let mut SinAngles: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        let mut CosAngles: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        XMVectorSinCos(&mut SinAngles, &mut CosAngles, HalfAngles);

        // TODO: PERFORMANCE XMVectorPermute
        // let P0: XMVECTOR = XMVectorPermute(SinAngles, CosAngles, XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X);
        // let Y0: XMVECTOR = XMVectorPermute(SinAngles, CosAngles, XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y);
        // let R0: XMVECTOR = XMVectorPermute(SinAngles, CosAngles, XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z);
        // let P1: XMVECTOR = XMVectorPermute(CosAngles, SinAngles, XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X);
        // let Y1: XMVECTOR = XMVectorPermute(CosAngles, SinAngles, XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y);
        // let R1: XMVECTOR = XMVectorPermute(CosAngles, SinAngles, XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z);

        // TODO: Delete note above after benchmarking
        let P0: XMVECTOR = <(XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X)>::XMVectorPermute(SinAngles, CosAngles);
        let Y0: XMVECTOR = <(XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y)>::XMVectorPermute(SinAngles, CosAngles);
        let R0: XMVECTOR = <(XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z)>::XMVectorPermute(SinAngles, CosAngles);
        let P1: XMVECTOR = <(XM_PERMUTE_0X, XM_PERMUTE_1X, XM_PERMUTE_1X, XM_PERMUTE_1X)>::XMVectorPermute(CosAngles, SinAngles);
        let Y1: XMVECTOR = <(XM_PERMUTE_1Y, XM_PERMUTE_0Y, XM_PERMUTE_1Y, XM_PERMUTE_1Y)>::XMVectorPermute(CosAngles, SinAngles);
        let R1: XMVECTOR = <(XM_PERMUTE_1Z, XM_PERMUTE_1Z, XM_PERMUTE_0Z, XM_PERMUTE_1Z)>::XMVectorPermute(CosAngles, SinAngles);

        let mut Q1: XMVECTOR = XMVectorMultiply(P1, Sign.v);
        let mut Q0: XMVECTOR = XMVectorMultiply(P0, Y0);
        Q1 = XMVectorMultiply(Q1, Y1);
        Q0 = XMVectorMultiply(Q0, R0);
        let Q: XMVECTOR = XMVectorMultiplyAdd(Q1, R1, Q0);

        return Q;
    }
}

/// Computes the rotation quaternion about a normal vector.
///
/// ## Parameters
///
/// `NormalAxis` Normal vector describing the axis of rotation.
///
/// `Angle` Angle of rotation in radians. Angles are measured clockwise when
/// looking along the rotation axis toward the origin.
///
/// ## Return value
///
/// Returns the rotation quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionRotationNormal>
#[inline]
pub fn XMQuaternionRotationNormal(
    NormalAxis: XMVECTOR,
    Angle: f32,
) -> FXMVECTOR
{
    #[cfg(any(_XM_NO_INTRINSICS_, _XM_ARM_NEON_INTRINSICS_))]
    unsafe {
        let N: XMVECTOR = XMVectorSelect(g_XMOne.v, NormalAxis, g_XMSelect1110.v);

        let mut SinV: f32 = 0.0;
        let mut CosV: f32 = 0.0;
        XMScalarSinCos(&mut SinV, &mut CosV, 0.5 * Angle);

        let Scale: XMVECTOR = XMVectorSet(SinV, SinV, SinV, CosV);
        return XMVectorMultiply(N, Scale);
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        let mut N: XMVECTOR = _mm_and_ps(NormalAxis, g_XMMask3.v);
        N = _mm_or_ps(N, g_XMIdentityR3.v);
        let mut Scale: XMVECTOR = _mm_set_ps1(0.5 * Angle);
        let mut vSine: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        let mut vCosine: XMVECTOR = mem::MaybeUninit::uninit().assume_init();
        XMVectorSinCos(&mut vSine, &mut vCosine, Scale);
        Scale = _mm_and_ps(vSine, g_XMMask3.v);
        vCosine = _mm_and_ps(vCosine, g_XMMaskW.v);
        Scale = _mm_or_ps(Scale, vCosine);
        N = _mm_mul_ps(N, Scale);
        return N;
    }
}

/// Computes a rotation quaternion about an axis.
///
/// ## Parameters
///
/// `Axis` 3D vector describing the axis of rotation.
///
/// `Angle` Angle of rotation in radians. Angles are measured clockwise when
/// looking along the rotation axis toward the origin.
///
/// ## Return value
///
/// Returns the rotation quaternion.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where
/// the `X`, `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// If Axis is a normalized vector, it is faster to use [`XMQuaternionRotationNormal`]
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionRotationAxis>
///
/// [`XMQuaternionRotationNormal`]: crate::quaternion::XMQuaternionRotationNormal
#[inline]
pub fn XMQuaternionRotationAxis(
    Axis: XMVECTOR,
    Angle: f32,
) -> FXMVECTOR
{
    debug_assert!(!XMVector3Equal(Axis, XMVectorZero()));
    debug_assert!(!XMVector3IsInfinite(Axis));

    let Normal: XMVECTOR = XMVector3Normalize(Axis);
    let Q: XMVECTOR = XMQuaternionRotationNormal(Normal, Angle);
    return Q;
}


/// Computes a rotation quaternion from a rotation matrix.
///
/// ## Parameters
///
/// `M` Rotation matrix.
///
/// ## Return value
///
/// Returns the rotation quaternion.
///
/// ## Remarks
///
/// This function only uses the upper 3x3 portion of the XMMATRIX. Note if the input matrix contains scales,
/// shears, or other non-rotation transformations in the upper 3x3 matrix, then the output of this function
/// is ill-defined.
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionRotationMatrix>
#[inline]
pub fn XMQuaternionRotationMatrix(
    M: XMMATRIX,
) -> FXMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut q: XMVECTORF32 = mem::MaybeUninit::uninit().assume_init();
        let r22: f32 = M.m[2][2];
        if (r22 <= 0.0)  // x^2 + y^2 >= z^2 + w^2
        {
            let dif10: f32 = M.m[1][1] - M.m[0][0];
            let omr22: f32 = 1.0 - r22;
            if (dif10 <= 0.0)  // x^2 >= y^2
            {
                let fourXSqr: f32 = omr22 - dif10;
                let inv4x: f32 = 0.5 / sqrtf(fourXSqr);
                q.f[0] = fourXSqr * inv4x;
                q.f[1] = (M.m[0][1] + M.m[1][0]) * inv4x;
                q.f[2] = (M.m[0][2] + M.m[2][0]) * inv4x;
                q.f[3] = (M.m[1][2] - M.m[2][1]) * inv4x;
            }
            else  // y^2 >= x^2
            {
                let fourYSqr: f32 = omr22 + dif10;
                let inv4y: f32 = 0.5 / sqrtf(fourYSqr);
                q.f[0] = (M.m[0][1] + M.m[1][0]) * inv4y;
                q.f[1] = fourYSqr * inv4y;
                q.f[2] = (M.m[1][2] + M.m[2][1]) * inv4y;
                q.f[3] = (M.m[2][0] - M.m[0][2]) * inv4y;
            }
        }
        else  // z^2 + w^2 >= x^2 + y^2
        {
            let sum10: f32 = M.m[1][1] + M.m[0][0];
            let opr22: f32 = 1.0 + r22;
            if (sum10 <= 0.0)  // z^2 >= w^2
            {
                let fourZSqr: f32 = opr22 - sum10;
                let inv4z: f32 = 0.5 / sqrtf(fourZSqr);
                q.f[0] = (M.m[0][2] + M.m[2][0]) * inv4z;
                q.f[1] = (M.m[1][2] + M.m[2][1]) * inv4z;
                q.f[2] = fourZSqr * inv4z;
                q.f[3] = (M.m[0][1] - M.m[1][0]) * inv4z;
            }
            else  // w^2 >= z^2
            {
                let fourWSqr: f32 = opr22 + sum10;
                let inv4w: f32 = 0.5 / sqrtf(fourWSqr);
                q.f[0] = (M.m[1][2] - M.m[2][1]) * inv4w;
                q.f[1] = (M.m[2][0] - M.m[0][2]) * inv4w;
                q.f[2] = (M.m[0][1] - M.m[1][0]) * inv4w;
                q.f[3] = fourWSqr * inv4w;
            }
        }
        return q.v;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        const XMPMMP: XMVECTORF32 = XMVECTORF32 { f: [  1.0, -1.0, -1.0,  1.0 ] };
        const XMMPMP: XMVECTORF32 = XMVECTORF32 { f: [ -1.0,  1.0, -1.0,  1.0 ] };
        const XMMMPP: XMVECTORF32 = XMVECTORF32 { f: [ -1.0, -1.0,  1.0,  1.0 ] };

        let r0: XMVECTOR = M.r[0];  // (r00, r01, r02, 0)
        let r1: XMVECTOR = M.r[1];  // (r10, r11, r12, 0)
        let r2: XMVECTOR = M.r[2];  // (r20, r21, r22, 0)

        // (r00, r00, r00, r00)
        let r00: XMVECTOR = XM_PERMUTE_PS!(r0, _MM_SHUFFLE(0, 0, 0, 0));
        // (r11, r11, r11, r11)
        let r11: XMVECTOR = XM_PERMUTE_PS!(r1, _MM_SHUFFLE(1, 1, 1, 1));
        // (r22, r22, r22, r22)
        let r22: XMVECTOR = XM_PERMUTE_PS!(r2, _MM_SHUFFLE(2, 2, 2, 2));

        // x^2 >= y^2 equivalent to r11 - r00 <= 0
        // (r11 - r00, r11 - r00, r11 - r00, r11 - r00)
        let r11mr00: XMVECTOR = _mm_sub_ps(r11, r00);
        let x2gey2: XMVECTOR = _mm_cmple_ps(r11mr00, g_XMZero.v);

        // z^2 >= w^2 equivalent to r11 + r00 <= 0
        // (r11 + r00, r11 + r00, r11 + r00, r11 + r00)
        let r11pr00: XMVECTOR = _mm_add_ps(r11, r00);
        let z2gew2: XMVECTOR = _mm_cmple_ps(r11pr00, g_XMZero.v);

        // x^2 + y^2 >= z^2 + w^2 equivalent to r22 <= 0
        let x2py2gez2pw2: XMVECTOR = _mm_cmple_ps(r22, g_XMZero.v);

        // (4*x^2, 4*y^2, 4*z^2, 4*w^2)
        let mut t0: XMVECTOR = XM_FMADD_PS!(XMPMMP.v, r00, g_XMOne.v);
        let mut t1: XMVECTOR = _mm_mul_ps(XMMPMP.v, r11);
        let mut t2: XMVECTOR = XM_FMADD_PS!(XMMMPP.v, r22, t0);
        let x2y2z2w2: XMVECTOR = _mm_add_ps(t1, t2);

        // (r01, r02, r12, r11)
        t0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 2, 2, 1));
        // (r10, r10, r20, r21)
        t1 = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(1, 0, 0, 0));
        // (r10, r20, r21, r10)
        t1 = XM_PERMUTE_PS!(t1, _MM_SHUFFLE(1, 3, 2, 0));
        // (4*x*y, 4*x*z, 4*y*z, unused)
        let xyxzyz: XMVECTOR = _mm_add_ps(t0, t1);

        // (r21, r20, r10, r10)
        t0 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(0, 0, 0, 1));
        // (r12, r12, r02, r01)
        t1 = _mm_shuffle_ps(r1, r0, _MM_SHUFFLE(1, 2, 2, 2));
        // (r12, r02, r01, r12)
        t1 = XM_PERMUTE_PS!(t1, _MM_SHUFFLE(1, 3, 2, 0));
        // (4*x*w, 4*y*w, 4*z*w, unused)
        let mut xwywzw: XMVECTOR = _mm_sub_ps(t0, t1);
        xwywzw = _mm_mul_ps(XMMPMP.v, xwywzw);

        // (4*x^2, 4*y^2, 4*x*y, unused)
        t0 = _mm_shuffle_ps(x2y2z2w2, xyxzyz, _MM_SHUFFLE(0, 0, 1, 0));
        // (4*z^2, 4*w^2, 4*z*w, unused)
        t1 = _mm_shuffle_ps(x2y2z2w2, xwywzw, _MM_SHUFFLE(0, 2, 3, 2));
        // (4*x*z, 4*y*z, 4*x*w, 4*y*w)
        t2 = _mm_shuffle_ps(xyxzyz, xwywzw, _MM_SHUFFLE(1, 0, 2, 1));

        // (4*x*x, 4*x*y, 4*x*z, 4*x*w)
        let tensor0: XMVECTOR = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
        // (4*y*x, 4*y*y, 4*y*z, 4*y*w)
        let tensor1: XMVECTOR = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 1, 1, 2));
        // (4*z*x, 4*z*y, 4*z*z, 4*z*w)
        let tensor2: XMVECTOR = _mm_shuffle_ps(t2, t1, _MM_SHUFFLE(2, 0, 1, 0));
        // (4*w*x, 4*w*y, 4*w*z, 4*w*w)
        let tensor3: XMVECTOR = _mm_shuffle_ps(t2, t1, _MM_SHUFFLE(1, 2, 3, 2));

        // Select the row of the tensor-product matrix that has the largest
        // magnitude.
        t0 = _mm_and_ps(x2gey2, tensor0);
        t1 = _mm_andnot_ps(x2gey2, tensor1);
        t0 = _mm_or_ps(t0, t1);
        t1 = _mm_and_ps(z2gew2, tensor2);
        t2 = _mm_andnot_ps(z2gew2, tensor3);
        t1 = _mm_or_ps(t1, t2);
        t0 = _mm_and_ps(x2py2gez2pw2, t0);
        t1 = _mm_andnot_ps(x2py2gez2pw2, t1);
        t2 = _mm_or_ps(t0, t1);

        // Normalize the row.  No division by zero is possible because the
        // quaternion is unit-length (and the row is a nonzero multiple of
        // the quaternion).
        t0 = XMVector4Length(t2);
        return _mm_div_ps(t2, t0);
    }
}

/// Computes an axis and angle of rotation about that axis for a given quaternion.
///
/// ## Parameters
///
/// `pAxis` Address of a 3D vector describing the axis of rotation for the quaternion `Q`.
///
/// `pAngle` Address of float describing the radian angle of rotation for the quaternion `Q`.
///
/// `Q` Quaternion to measure.
///
/// ## Return value
///
/// None.
///
/// ## Remarks
///
/// The DirectXMath quaternion functions use an XMVECTOR 4-vector to represent quaternions, where the `X`,
/// `Y`, and `Z` components are the vector part and the `W` component is the scalar part.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMQuaternionToAxisAngle>
#[inline]
pub fn XMQuaternionToAxisAngle(
    pAxis: &mut FXMVECTOR,
    pAngle: &mut f32,
    Q: FXMVECTOR,
)
{
    *pAxis = Q;
    *pAngle = 2.0 * XMScalarACos(XMVectorGetW(Q));
}

/// Determines if two planes are equal.
///
/// ## Parameters
///
/// `P1` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `P2` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// Returns `true` if the two planes are equal and `false` otherwise.
///
/// ## Remarks
///
/// The following pseudocode demonstrates the operation of the function.
///
/// ```text
/// return (P1.x == P2.x && P1.y == P2.y && P1.z == P2.z && P1.w == P2.w);
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneEqual>
#[inline]
pub fn XMPlaneEqual(
    P1: FXMVECTOR,
    P2: FXMVECTOR,
) -> bool
{
    return XMVector4Equal(P1, P2);
}


/// Determines whether two planes are nearly equal.
///
/// ## Parameters
///
/// `P1` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `P2` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `Epsilon` An XMVECTOR that gives the component-wise tolerance to use when comparing `P1` and `P2`.
///
/// ## Return value
///
/// Returns `true` if `P1` is nearly equal to `P2` and `false` otherwise.
///
/// ## Remarks
///
/// The XMPlaneNearEqual function normalizes the `P1` and `P2` parameters before passing them, and the `Epsilon`
/// parameter, to the [`XMVector4NearEqual`] function. For more information about how the calculation is performed,
/// see the [`XMVector4NearEqual`] function.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneNearEqual>
///
/// [`XMVector4NearEqual`]: crate::vector::XMVector4NearEqual
#[inline]
pub fn XMPlaneNearEqual(
    P1: FXMVECTOR,
    P2: FXMVECTOR,
    Epsilon: FXMVECTOR,
) -> bool
{
    let NP1: XMVECTOR = XMPlaneNormalize(P1);
    let NP2: XMVECTOR = XMPlaneNormalize(P2);
    return XMVector4NearEqual(NP1, NP2, Epsilon);
}

/// Determines if two planes are equal.
///
/// ## Parameters
///
/// `P1` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `P2` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// Returns `true` if the two planes are unequal and `false` otherwise.
///
/// ## Remarks
///
/// The following pseudocode demonstrates the operation of the function.
///
/// ```text
/// return (P1.x != P2.x || P1.y != P2.y || P1.z != P2.z || P1.w != P2.w);
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneNotEqual>
#[inline]
pub fn XMPlaneNotEqual(
    P1: FXMVECTOR,
    P2: FXMVECTOR,
) -> bool
{
    return XMVector4NotEqual(P1, P2);
}

/// Tests whether any of the coefficients of a plane is a NaN.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// Returns `true` if any of the coefficients of the plane is a `NaN`, and `false` otherwise.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneIsNaN>
#[inline]
pub fn XMPlaneIsNaN(
    P: FXMVECTOR,
) -> bool
{
    return XMVector4IsNaN(P);
}

/// Tests whether any of the coefficients of a plane is positive or negative infinity.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// Returns `true` if any of the coefficients of the plane is positive or negative infinity, and `false` otherwise.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneIsInfinite>
#[inline]
pub fn XMPlaneIsInfinite(
    P: FXMVECTOR,
) -> bool
{
    return XMVector4IsInfinite(P);
}

/// Calculates the dot product between an input plane and a 4D vector.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `V` 4D vector to use in the dot product.
///
/// ## Return value
///
/// Returns the dot product of `P` and `V` replicated into each of the four components of the returned XMVECTOR.
///
/// ## Remarks
///
/// The XMPlaneDot function is useful for determining the plane's relationship with a homogeneous coordinate.
/// For example, this function can be used to determine if a particular coordinate is on a particular plane,
/// or on which side of a particular plane a particular coordinate lies.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneDot>
#[inline]
pub fn XMPlaneDot(
    P: FXMVECTOR,
    V: FXMVECTOR,
) -> XMVECTOR
{
    return XMVector4Dot(P, V);
}

/// Calculates the dot product between an input plane and a 3D vector.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `V` 3D vector to use in the dot product. The `w` component of `V` is always treated as if is `1.0`.
///
/// ## Return value
///
/// Returns the dot product between `P` and `V` replicated into each of the four components of the returned
/// XMVECTOR.
///
/// ## Remarks
///
/// This function can be useful in finding the signed distance from a point to a plane. The following pseudocode
/// demonstrates the operation of the function.
///
/// ```text
/// XMVECTOR vectorOut;
///
/// vectorOut.x = P.x * V.x + P.y * V.y + P.z * V.z + P.w * 1.0f;
/// vectorOut.y = P.x * V.x + P.y * V.y + P.z * V.z + P.w * 1.0f;
/// vectorOut.z = P.x * V.x + P.y * V.y + P.z * V.z + P.w * 1.0f;
/// vectorOut.w = P.x * V.x + P.y * V.y + P.z * V.z + P.w * 1.0f;
///
/// return vectorOut;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneDotCoord>
#[inline]
pub fn XMPlaneDotCoord(
    P: FXMVECTOR,
    V: FXMVECTOR,
) -> XMVECTOR
{
    // Result = P[0] * V[0] + P[1] * V[1] + P[2] * V[2] + P[3]

    unsafe {
        let V3: XMVECTOR = XMVectorSelect(g_XMOne.v, V, g_XMSelect1110.v);
        let Result: XMVECTOR = XMVector4Dot(P, V3);
        return Result;
    }
}


/// Calculates the dot product between the normal vector of a plane and a 3D vector.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `V` 3D vector to use in the dot product. The `w` component of `V` is always treated as if is `0.0`.
///
/// ## Return value
///
/// Returns the dot product between the normal vector of the plane and `V` replicated into each of the four
/// components of the returned XMVECTOR.
///
/// ## Remarks
///
/// This function is useful for calculating the angle between the normal vector of the plane, and another
/// normal vector. The following pseudocode demonstrates the operation of the function.
///
/// ```text
/// XMVECTOR vectorOut;
///
/// vectorOut.x = P.x * V.x + P.y * V.y + P.z * V.z;
/// vectorOut.y = P.x * V.x + P.y * V.y + P.z * V.z;
/// vectorOut.z = P.x * V.x + P.y * V.y + P.z * V.z;
/// vectorOut.w = P.x * V.x + P.y * V.y + P.z * V.z;
///
/// return vectorOut;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneDotNormal>
#[inline]
pub fn XMPlaneDotNormal(
    P: FXMVECTOR,
    V: FXMVECTOR,
) -> XMVECTOR
{
    return XMVector3Dot(P, V);
}

/// Estimates the coefficients of a plane so that coefficients of x, y, and z form a unit normal vector.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// Returns an estimation of the normalized plane as a 4D vector whose components are the plane coefficients
/// (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Remarks
///
/// `Est` functions offer increased performance at the expense of reduced accuracy. `Est` functions are appropriate
/// for non-critical calculations where accuracy can be sacrificed for speed. The exact amount of lost accuracy
/// and speed increase are platform dependent.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneNormalizeEst>
#[inline]
pub fn XMPlaneNormalizeEst(
    P: FXMVECTOR,
) -> XMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    {
        let Result: XMVECTOR = XMVector3ReciprocalLengthEst(P);
        return XMVectorMultiply(P, Result);
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // Perform the dot product
        let mut vDot: XMVECTOR = _mm_mul_ps(P, P);
        // x=Dot.y, y=Dot.z
        let mut vTemp: XMVECTOR = XM_PERMUTE_PS!(vDot, _MM_SHUFFLE(2, 1, 2, 1));
        // Result.x = x+y
        vDot = _mm_add_ss(vDot, vTemp);
        // x=Dot.z
        vTemp = XM_PERMUTE_PS!(vTemp, _MM_SHUFFLE(1, 1, 1, 1));
        // Result.x = (x+y)+z
        vDot = _mm_add_ss(vDot, vTemp);
        // Splat x
        vDot = XM_PERMUTE_PS!(vDot, _MM_SHUFFLE(0, 0, 0, 0));
        // Get the reciprocal
        vDot = _mm_rsqrt_ps(vDot);
        // Get the reciprocal
        vDot = _mm_mul_ps(vDot, P);
        return vDot;
    }
}

/// Normalizes the coefficients of a plane so that coefficients of x, y, and z form a unit normal vector.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// Returns the normalized plane as a 4D vector whose components are the plane coefficients (`A`, `B`, `C`, `D`)
/// for the plane equation  `Ax+By+Cz+D=0`.
///
/// ## Remarks
///
/// The following pseudocode demonstrates the operation of the function:
///
/// ```text
/// XMVECTOR Result;
///
/// float LengthSq = P.x * P.x + P.y * P.y + P.z * P.z;
///
/// float ReciprocalLength = 1.0f / sqrt(LengthSq);
/// Result.x = P.x * ReciprocalLength;
/// Result.y = P.y * ReciprocalLength;
/// Result.z = P.z * ReciprocalLength;
/// Result.w = P.w * ReciprocalLength;
///
/// return Result;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneNormalize>
#[inline]
pub fn XMPlaneNormalize(
    P: FXMVECTOR,
) -> XMVECTOR
{
    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut fLengthSq: f32 = sqrtf((P.vector4_f32[0] * P.vector4_f32[0]) + (P.vector4_f32[1] * P.vector4_f32[1]) + (P.vector4_f32[2] * P.vector4_f32[2]));
        // Prevent divide by zero
        if (fLengthSq > 0.0)
        {
            fLengthSq = 1.0 / fLengthSq;
        }
        let vResult = XMVECTORF32 { f: [
                P.vector4_f32[0] * fLengthSq,
                P.vector4_f32[1] * fLengthSq,
                P.vector4_f32[2] * fLengthSq,
                P.vector4_f32[3] * fLengthSq
        ] };
        return vResult.v;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    unsafe {
        unimplemented!()
    }

    #[cfg(_XM_SSE4_INTRINSICS_)]
    unsafe {
        let mut vLengthSq: XMVECTOR = _mm_dp_ps(P, P, 0x7f);
        // Prepare for the division
        let mut vResult: XMVECTOR = _mm_sqrt_ps(vLengthSq);
        // Failsafe on zero (Or epsilon) length planes
        // If the length is infinity, set the elements to zero
        vLengthSq = _mm_cmpneq_ps(vLengthSq, g_XMInfinity.v);
        // Reciprocal mul to perform the normalization
        vResult = _mm_div_ps(P, vResult);
        // Any that are infinity, set to zero
        vResult = _mm_and_ps(vResult, vLengthSq);
        return vResult;
    }

    #[cfg(all(_XM_SSE_INTRINSICS_, not(_XM_SSE4_INTRINSICS_)))]
    unsafe {
        let mut vLengthSq: XMVECTOR = _mm_dp_ps(P, P, 0x7f);
        // Prepare for the division
        let mut vResult: XMVECTOR = _mm_sqrt_ps(vLengthSq);
        // Failsafe on zero (Or epsilon) length planes
        // If the length is infinity, set the elements to zero
        vLengthSq = _mm_cmpneq_ps(vLengthSq, g_XMInfinity.v);
        // Reciprocal mul to perform the normalization
        vResult = _mm_div_ps(P, vResult);
        // Any that are infinity, set to zero
        vResult = _mm_and_ps(vResult, vLengthSq);
        return vResult;
    }
}

/// Finds the intersection between a plane and a line.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `LinePoint1` 3D vector describing the first point on the line.
///
/// `LinePoint2` 3D vector describing the second point on the line.
///
/// ## Return value
///
/// Returns the intersection of the plane P and the line defined by `LinePoint1` and `LinePoint2`. If the line
/// is parallel to the plane, all components of the returned vector are equal to `QNaN`.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneIntersectLine>
#[inline]
pub fn XMPlaneIntersectLine(
    P: FXMVECTOR,
    LinePoint1: FXMVECTOR,
    LinePoint2: FXMVECTOR,
) -> XMVECTOR
{
    unsafe {
        let V1: XMVECTOR = XMVector3Dot(P, LinePoint1);
        let V2: XMVECTOR = XMVector3Dot(P, LinePoint2);
        let D: XMVECTOR = XMVectorSubtract(V1, V2);

        let mut VT: XMVECTOR = XMPlaneDotCoord(P, LinePoint1);
        VT = XMVectorDivide(VT, D);

        let mut Point: XMVECTOR = XMVectorSubtract(LinePoint2, LinePoint1);
        Point = XMVectorMultiplyAdd(Point, VT, LinePoint1);

        // const
        let Zero: XMVECTOR = XMVectorZero();
        let Control: XMVECTOR = XMVectorNearEqual(D, Zero, g_XMEpsilon.v);

        return XMVectorSelect(Point, g_XMQNaN.v, Control);
    }
}

/// Finds the intersection of two planes.
///
/// ## Parameters
///
/// `pLinePoint1` Address of a 3D vector describing one point on the line of intersection. See remarks.
///
/// `pLinePoint2` Address of a 3D vector describing a second point on the line of intersection. See remarks.
///
/// `P1` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `P2` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Return value
///
/// None.
///
/// ## Remarks
///
/// If the planes are parallel to one another, all components of the returned point vectors are equal to
/// QNaN.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneIntersectPlane>
#[inline]
pub fn XMPlaneIntersectPlane(
    pLinePoint1: &mut FXMVECTOR,
    pLinePoint2: &mut FXMVECTOR,
    P1: FXMVECTOR,
    P2: FXMVECTOR,
)
{
    unsafe {
        let V1: XMVECTOR = XMVector3Cross(P2, P1);

        let LengthSq: XMVECTOR = XMVector3LengthSq(V1);

        let V2: XMVECTOR = XMVector3Cross(P2, V1);

        let P1W: XMVECTOR = XMVectorSplatW(P1);
        let mut Point: XMVECTOR = XMVectorMultiply(V2, P1W);

        let V3: XMVECTOR = XMVector3Cross(V1, P1);

        let P2W: XMVECTOR = XMVectorSplatW(P2);
        Point = XMVectorMultiplyAdd(V3, P2W, Point);

        let LinePoint1: XMVECTOR = XMVectorDivide(Point, LengthSq);

        let LinePoint2: XMVECTOR = XMVectorAdd(LinePoint1, V1);

        let Control: XMVECTOR = XMVectorLessOrEqual(LengthSq, g_XMEpsilon.v);
        *pLinePoint1 = XMVectorSelect(LinePoint1, g_XMQNaN.v, Control);
        *pLinePoint2 = XMVectorSelect(LinePoint2, g_XMQNaN.v, Control);
    }
}

/// Transforms a plane by a given matrix.
///
/// ## Parameters
///
/// `P` XMVECTOR describing the plane coefficients (`A`, `B`, `C`, `D`) for the plane equation `Ax+By+Cz+D=0`.
///
/// `M` Transformation matrix.
///
/// ## Return value
///
/// Returns the transformed plane as a 4D vector whose components are the plane coefficients (`A`, `B`, `C`, `D`)
/// for the plane equation `Ax+By+Cz+D=0`.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneTransform>
#[inline]
pub fn XMPlaneTransform(
    P: FXMVECTOR,
    M: FXMMATRIX,
) -> XMVECTOR
{
    unsafe {
        let W: XMVECTOR = XMVectorSplatW(P);
        let Z: XMVECTOR = XMVectorSplatZ(P);
        let Y: XMVECTOR = XMVectorSplatY(P);
        let X: XMVECTOR = XMVectorSplatX(P);

        let mut Result: XMVECTOR = XMVectorMultiply(W, M.r[3]);
        Result = XMVectorMultiplyAdd(Z, M.r[2], Result);
        Result = XMVectorMultiplyAdd(Y, M.r[1], Result);
        Result = XMVectorMultiplyAdd(X, M.r[0], Result);
        return Result;
    }
}

// TODO: XMPlaneTransformStream

/// Computes the equation of a plane constructed from a point in the plane and a normal vector.
///
/// ## Parameters
///
/// `Point` 3D vector describing a point in the plane.
///
/// `Normal` 3D vector normal to the plane.
///
/// ## Return value
///
/// Returns a vector whose components are the coefficients of the plane (`A`, `B`, `C`, `D`) for the plane equation
/// `Ax+By+Cz+D=0`.
///
/// ## Remarks
///
/// The following pseudocode demonstrates the operation of the function:
///
/// ```text
/// XMVECTOR Result;
///
/// Result.x = Normal.x;
/// Result.y = Normal.y;
/// Result.z = Normal.z;
/// Result.w = -(Point.x * Normal.x + Point.y * Normal.y + Point.z * Normal.z);
///
/// return Result;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneFromPointNormal>
#[inline]
pub fn XMPlaneFromPointNormal(
    Point: FXMVECTOR,
    Normal: FXMVECTOR,
) -> XMVECTOR
{
    unsafe {
        let mut W: XMVECTOR = XMVector3Dot(Point, Normal);
        W = XMVectorNegate(W);
        return XMVectorSelect(W, Normal, g_XMSelect1110.v);
    }
}

/// Computes the equation of a plane constructed from three points in the plane.
///
/// ## Parameters
///
/// `Point1` 3D vector describing a point in the plane.
///
/// `Point2` 3D vector describing a point in the plane.
///
/// `Point3` 3D vector describing a point in the plane.
///
/// ## Return value
///
/// Returns a vector whose components are the coefficients of the plane (`A`, `B`, `C`, `D`) for the plane equation
/// `Ax+By+Cz+D=0`.
///
/// ## Remarks
///
/// The following pseudocode demonstrates the operation of the function:
///
/// ```text
/// XMVECTOR Result;
/// XMVECTOR N;
/// XMVECTOR D;
///
/// XMVECTOR V21 = XMVectorSubtract(Point1, Point2);
/// XMVECTOR V31 = XMVectorSubtract(Point1, Point3);
///
/// N = XMVector3Cross(V21, V31);
/// N = XMVector3Normalize(N);
///
/// D = XMPlaneDotNormal(N, Point1);
///
/// Result.x = N.x;
/// Result.y = N.y;
/// Result.z = N.z;
/// Result.w = -D.w;
///
/// return Result;
/// ```
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMPlaneFromPoints>
#[inline]
pub fn XMPlaneFromPoints(
    Point1: FXMVECTOR,
    Point2: FXMVECTOR,
    Point3: FXMVECTOR,
) -> XMVECTOR
{
    unsafe {
        let V21: XMVECTOR = XMVectorSubtract(Point1, Point2);
        let V31: XMVECTOR = XMVectorSubtract(Point1, Point3);

        let mut N: XMVECTOR = XMVector3Cross(V21, V31);
        N = XMVector3Normalize(N);

        let mut D: XMVECTOR = XMPlaneDotNormal(N, Point1);
        D = XMVectorNegate(D);

        let Result: XMVECTOR = XMVectorSelect(D, N, g_XMSelect1110.v);

        return Result;
    }
}

// TODO: XMColorEqual
// TODO: XMColorNotEqual
// TODO: XMColorGreater
// TODO: XMColorGreaterOrEqual
// TODO: XMColorLess
// TODO: XMColorLessOrEqual
// TODO: XMColorIsNaN
// TODO: XMColorIsInfinite
// TODO: XMColorNegative
// TODO: XMColorModulate
// TODO: XMColorAdjustSaturation
// TODO: XMColorAdjustContrast
// TODO: XMColorRGBToHSL
// TODO: XMColorHue2Clr
// TODO: XMColorHSLToRGB
// TODO: XMColorRGBToHSV
// TODO: XMColorHSVToRGB
// TODO: XMColorRGBToYUV
// TODO: XMColorYUVToRGB
// TODO: XMColorRGBToYUV_HD
// TODO: XMColorYUVToRGB_HD
// TODO: XMColorRGBToXYZ
// TODO: XMColorXYZToRGB
// TODO: XMColorXYZToSRGB
// TODO: XMColorSRGBToXYZ
// TODO: XMColorRGBToSRGB
// TODO: XMColorSRGBToRGB

// TODO: XMVerifyCPUSupport (NOTE: __cpuid)

/// Calculates the Fresnel term for unpolarized light.
///
/// ## Parameters
///
/// `CosIncidentAngle` Vector consisting of the cosines of the incident angles.
///
/// `RefractionIndex` Vector consisting of the refraction indices of the materials corresponding to the incident angles.
///
/// ## Return value
///
/// Returns a vector containing the Fresnel term of each component.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMFresnelTerm>
#[inline]
pub fn XMFresnelTerm(
    CosIncidentAngle: FXMVECTOR,
    RefractionIndex: FXMVECTOR,
) -> FXMVECTOR
{
    debug_assert!(!XMVector4IsInfinite(CosIncidentAngle));

    // Result = 0.5f * (g - c)^2 / (g + c)^2 * ((c * (g + c) - 1)^2 / (c * (g - c) + 1)^2 + 1) where
    // c = CosIncidentAngle
    // g = sqrt(c^2 + RefractionIndex^2 - 1)

    #[cfg(_XM_NO_INTRINSICS_)]
    unsafe {
        let mut G: XMVECTOR = XMVectorMultiplyAdd(RefractionIndex, RefractionIndex, g_XMNegativeOne.v);
        G = XMVectorMultiplyAdd(CosIncidentAngle, CosIncidentAngle, G);
        G = XMVectorAbs(G);
        G = XMVectorSqrt(G);

        let S: XMVECTOR = XMVectorAdd(G, CosIncidentAngle);
        let D: XMVECTOR = XMVectorSubtract(G, CosIncidentAngle);

        let mut V0: XMVECTOR = XMVectorMultiply(D, D);
        let mut V1: XMVECTOR = XMVectorMultiply(S, S);
        V1 = XMVectorReciprocal(V1);
        V0 = XMVectorMultiply(g_XMOneHalf.v, V0);
        V0 = XMVectorMultiply(V0, V1);

        let mut V2: XMVECTOR = XMVectorMultiplyAdd(CosIncidentAngle, S, g_XMNegativeOne.v);
        let mut V3: XMVECTOR = XMVectorMultiplyAdd(CosIncidentAngle, D, g_XMOne.v);
        V2 = XMVectorMultiply(V2, V2);
        V3 = XMVectorMultiply(V3, V3);
        V3 = XMVectorReciprocal(V3);
        V2 = XMVectorMultiplyAdd(V2, V3, g_XMOne.v);

        let mut Result: XMVECTOR = XMVectorMultiply(V0, V2);

        Result = XMVectorSaturate(Result);

        return Result;
    }

    #[cfg(_XM_ARM_NEON_INTRINSICS_)]
    {
        unimplemented!()
    }

    #[cfg(_XM_SSE_INTRINSICS_)]
    unsafe {
        // G = sqrt(abs((RefractionIndex^2-1) + CosIncidentAngle^2))
        let mut G: XMVECTOR = _mm_mul_ps(RefractionIndex, RefractionIndex);
        let mut vTemp: XMVECTOR = _mm_mul_ps(CosIncidentAngle, CosIncidentAngle);
        G = _mm_sub_ps(G, g_XMOne.v);
        vTemp = _mm_add_ps(vTemp, G);
        // max((0-vTemp),vTemp) == abs(vTemp)
        // The abs is needed to deal with refraction and cosine being zero
        G = _mm_setzero_ps();
        G = _mm_sub_ps(G, vTemp);
        G = _mm_max_ps(G, vTemp);
        // Last operation, the sqrt()
        G = _mm_sqrt_ps(G);

        // Calc G-C and G+C
        let mut GAddC: XMVECTOR = _mm_add_ps(G, CosIncidentAngle);
        let mut GSubC: XMVECTOR = _mm_sub_ps(G, CosIncidentAngle);
        // Perform the term (0.5f *(g - c)^2) / (g + c)^2
        let mut vResult: XMVECTOR = _mm_mul_ps(GSubC, GSubC);
        vTemp = _mm_mul_ps(GAddC, GAddC);
        vResult = _mm_mul_ps(vResult, g_XMOneHalf.v);
        vResult = _mm_div_ps(vResult, vTemp);
        // Perform the term ((c * (g + c) - 1)^2 / (c * (g - c) + 1)^2 + 1)
        GAddC = _mm_mul_ps(GAddC, CosIncidentAngle);
        GSubC = _mm_mul_ps(GSubC, CosIncidentAngle);
        GAddC = _mm_sub_ps(GAddC, g_XMOne.v);
        GSubC = _mm_add_ps(GSubC, g_XMOne.v);
        GAddC = _mm_mul_ps(GAddC, GAddC);
        GSubC = _mm_mul_ps(GSubC, GSubC);
        GAddC = _mm_div_ps(GAddC, GSubC);
        GAddC = _mm_add_ps(GAddC, g_XMOne.v);
        // Multiply the two term parts
        vResult = _mm_mul_ps(vResult, GAddC);
        // Clamp to 0.0 - 1.0f
        vResult = _mm_max_ps(vResult, g_XMZero.v);
        vResult = _mm_min_ps(vResult, g_XMOne.v);
        return vResult;
    }
}

/// Determines if two floating-point values are nearly equal.
///
/// ## Parameters
///
/// `S1` First floating-point value to compare.
///
/// `S2` Second floating-point value to compare.
///
/// `Epsilon` Tolerance to use when comparing `S1` and `S2`.
///
/// ## Return value
///
/// Returns `true` if the absolute value of the difference between `S1` and `S2` is less than or equal to `Epsilon`.
/// Returns `false` otherwise.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarNearEqual>
#[inline]
pub fn XMScalarNearEqual(
    S1: f32,
    S2: f32,
    Epsilon: f32
) -> bool
{
    let Delta: f32 = S1 - S2;
    return (fabsf(Delta) <= Epsilon);
}

/// Computes an angle between `-XM_PI` and `XM_PI`.
///
/// ## Parameters
///
/// `Value` float value describing the radian angle.
///
/// ## Return value
///
/// Returns an angle greater than or equal to `-XM_PI` and less than `XM_PI` that is congruent to `Value` modulo
/// `2 PI`.
///
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarModAngle>
#[inline]
pub fn XMScalarModAngle(
    Angle: f32
) -> f32
{
    // Note: The modulo is performed with unsigned math only to work
    // around a precision error on numbers that are close to PI

    // Normalize the range from 0.0f to XM_2PI
    let Angle = Angle + XM_PI;
    // Perform the modulo, unsigned
    let mut fTemp: f32 = fabsf(Angle);
    fTemp = fTemp - (XM_2PI * ((fTemp / XM_2PI) as i32) as f32);
    // Restore the number to the range of -XM_PI to XM_PI-epsilon
    fTemp = fTemp - XM_PI;
    // If the modulo'd value was negative, restore negation
    if (Angle < 0.0)
    {
        fTemp = -fTemp;
    }
    return fTemp;
}

/// Computes the sine of a radian angle.
///
/// # Remarks
///
/// This function uses a 11-degree minimax approximation.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarSin>
#[inline]
pub fn XMScalarSin(
    Value: f32
) -> f32
{
    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    let mut quotient: f32 = XM_1DIV2PI * Value;
    if (Value >= 0.0)
    {
        quotient = ((quotient + 0.5) as i32) as f32;
    }
    else
    {
        quotient = ((quotient - 0.5) as i32) as f32;
    }
    let mut y: f32 = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
    if (y > XM_PIDIV2)
    {
        y = XM_PI - y;
    }
    else if (y < -XM_PIDIV2)
    {
        y = -XM_PI - y;
    }

    // 11-degree minimax approximation
    let y2: f32 = y * y;
    return (((((-2.3889859e-08 * y2 + 2.7525562e-06) * y2 - 0.00019840874) * y2 + 0.0083333310) * y2 - 0.16666667) * y2 + 1.0) * y;
}

/// Estimates the sine of a radian angle.
///
/// # Remarks
///
/// This function uses a 7-degree minimax approximation.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarSinEst>
#[inline]
pub fn XMScalarSinEst(
    Value: f32
) -> f32
{
    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    let mut quotient: f32 = XM_1DIV2PI * Value;
    if (Value >= 0.0)
    {
        quotient = ((quotient + 0.5) as i32) as f32;
    }
    else
    {
        quotient = ((quotient - 0.5) as i32) as f32;
    }
    let mut y: f32 = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
    if (y > XM_PIDIV2)
    {
        y = XM_PI - y;
    }
    else if (y < -XM_PIDIV2)
    {
        y = -XM_PI - y;
    }

    // 7-degree minimax approximation
    let y2: f32 = y * y;
    return (((-0.00018524670 * y2 + 0.0083139502) * y2 - 0.16665852) * y2 + 1.0) * y;
}

/// Computes the cosine of a radian angle.
///
/// # Remarks
///
/// This function uses a 10-degree minimax approximation.
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarCos>
#[inline]
pub fn XMScalarCos(
    Value: f32
) -> f32
{
    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    let mut quotient: f32 = XM_1DIV2PI * Value;
    if (Value >= 0.0)
    {
        quotient = ((quotient + 0.5) as i32) as f32;
    }
    else
    {
        quotient = ((quotient - 0.5) as i32) as f32;
    }
    let mut y: f32 = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with cos(y) = sign*cos(x).
    let sign: f32;
    if (y > XM_PIDIV2)
    {
        y = XM_PI - y;
        sign = -1.0;
    }
    else if (y < -XM_PIDIV2)
    {
        y = -XM_PI - y;
        sign = -1.0;
    }
    else
    {
        sign =  1.0;
    }

    // 10-degree minimax approximation
    let y2: f32 = y * y;
    let p: f32 = ((((-2.6051615e-07 * y2 + 2.4760495e-05) * y2 - 0.0013888378) * y2 + 0.041666638) * y2 - 0.5) * y2 + 1.0;
    return sign * p;
}

/// Estimates the cosine of a radian angle.
///
/// ## Parameters
///
/// `Value` float value describing the radian angle.
///
/// ## Return value
///
/// Returns the cosine of Value.
///
/// ## Remarks
///
/// `Est` functions offer increased performance at the expense of reduced accuracy. `Est` functions are appropriate
/// for non-critical calculations where accuracy can be sacrificed for speed. The exact amount of lost accuracy
/// and speed increase are platform dependent.
///
/// This function uses a 6-degree minimax approximation.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarCosEst>
#[inline]
pub fn XMScalarCosEst(
    Value: f32
) -> f32
{
    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    let mut quotient: f32 = XM_1DIV2PI * Value;
    if (Value >= 0.0)
    {
        quotient = ((quotient + 0.5) as i32) as f32;
    }
    else
    {
        quotient = ((quotient - 0.5) as i32) as f32;
    }
    let mut y: f32 = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with cos(y) = sign*cos(x).
    let sign: f32;
    if (y > XM_PIDIV2)
    {
        y = XM_PI - y;
        sign = -1.0;
    }
    else if (y < -XM_PIDIV2)
    {
        y = -XM_PI - y;
        sign = -1.0;
    }
    else
    {
        sign =  1.0;
    }

    // 6-degree minimax approximation
    let y2: f32 = y * y;
    let p: f32 = ((-0.0012712436 * y2 + 0.041493919) * y2 - 0.49992746) * y2 + 1.0;
    return sign * p;
}

/// Computes both the sine and cosine of a radian angle.
///
/// ## Parameters
///
/// `pSin` Address of a float that will contain the result of the sine of Value.
///
/// `pCos` Address of a float that will contain the result of the cosine of Value.
///
/// `Value` float value describing the radian angle.
///
/// ## Return value
///
/// None
///
/// ## Remarks
///
/// This function uses a 11-degree minimax approximation for sine; 10-degree for cosine.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarSinCos>
#[inline]
pub fn XMScalarSinCos(
    pSin: &mut f32,
    pCos: &mut f32,
    Value: f32,
)
{
    // assert(pSin);
    // assert(pCos);

    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    let mut quotient: f32 = XM_1DIV2PI * Value;
    if (Value >= 0.0)
    {
        quotient = ((quotient + 0.5) as i32) as f32;
    }
    else
    {
        quotient = ((quotient - 0.5) as i32) as f32;
    }
    let mut y: f32 = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
    let sign: f32;
    if (y > XM_PIDIV2)
    {
        y = XM_PI - y;
        sign = -1.0;
    }
    else if (y < -XM_PIDIV2)
    {
        y = -XM_PI - y;
        sign = -1.0;
    }
    else
    {
        sign = 1.0; // +1.0
    }

    let y2: f32 = y * y;

    // 11-degree minimax approximation
    *pSin = (((((-2.3889859e-08 * y2 + 2.7525562e-06) * y2 - 0.00019840874) * y2 + 0.0083333310) * y2 - 0.16666667) * y2 + 1.0) * y;

    // 10-degree minimax approximation
    let p: f32 = ((((-2.6051615e-07 * y2 + 2.4760495e-05) * y2 - 0.0013888378) * y2 + 0.041666638) * y2 - 0.5) * y2 + 1.0;
    *pCos = sign * p;
}

/// Estimates both the sine and cosine of a radian angle.
///
/// ## Parameters
///
/// `pSin` Address of a float that will contain the result of the sine of Value.
///
/// `pCos` Address of a float that will contain the result of the cosine of Value.
///
/// `Value` float value describing the radian angle.
///
/// ## Return value
///
/// None
///
/// ## Remarks
///
/// `Est` functions offer increased performance at the expense of reduced accuracy. `Est` functions are appropriate
/// for non-critical calculations where accuracy can be sacrificed for speed. The exact amount of lost accuracy
/// and speed increase are platform dependent.
///
/// This function uses a 7-degree minimax approximation for sine; 6-degree for cosine.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarSinCosEst>
#[inline]
pub fn XMScalarSinCosEst(
    pSin: &mut f32,
    pCos: &mut f32,
    Value: f32,
)
{
    // assert(pSin);
    // assert(pCos);

    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    let mut quotient: f32 = XM_1DIV2PI * Value;
    if (Value >= 0.0)
    {
        quotient = ((quotient + 0.5) as i32) as f32;
    }
    else
    {
        quotient = ((quotient - 0.5) as i32) as f32;
    }
    let mut y: f32 = Value - XM_2PI * quotient;

    // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
    let sign: f32;
    if (y > XM_PIDIV2)
    {
        y = XM_PI - y;
        sign = -1.0;
    }
    else if (y < -XM_PIDIV2)
    {
        y = -XM_PI - y;
        sign = -1.0;
    }
    else
    {
        sign =  1.0;
    }

    let y2: f32 = y * y;

    // 7-degree minimax approximation
    *pSin = (((-0.00018524670 * y2 + 0.0083139502) * y2 - 0.16665852) * y2 + 1.0) * y;

    // 6-degree minimax approximation
    let p: f32 = ((-0.0012712436 * y2 + 0.041493919) * y2 - 0.49992746) * y2 + 1.0;
    *pCos = sign * p;
}

/// Computes the arcsine of a floating-point number.
///
/// ## Parameters
///
/// `Value` float value between `-1.0` and `1.0`.
///
/// ## Return value
///
/// Returns the inverse sine of Value.
///
/// ## Remarks
///
/// This function uses a 7-degree minimax approximation
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarASin>
#[inline]
pub fn XMScalarASin(
    Value: f32,
) -> f32
{
    // Clamp input to [-1,1].
    let nonnegative: bool = (Value >= 0.0);
    let x: f32 = fabsf(Value);
    let mut omx: f32 = 1.0 - x;
    if (omx < 0.0)
    {
        omx = 0.0;
    }
    let root: f32 = sqrtf(omx);

    // 7-degree minimax approximation
    let mut result: f32 = ((((((-0.0012624911 * x + 0.0066700901) * x - 0.0170881256) * x + 0.0308918810) * x - 0.0501743046) * x + 0.0889789874) * x - 0.2145988016) * x + 1.5707963050;
    result *= root;  // acos(|x|)

    // acos(x) = pi - acos(-x) when x < 0, asin(x) = pi/2 - acos(x)
    return (if nonnegative { XM_PIDIV2 - result } else { result - XM_PIDIV2 });
}


/// Estimates the arcsine of a floating-point number.
///
/// ## Parameters
///
/// `Value` float value between `-1.0` and `1.0`.
///
/// ## Return value
///
/// Returns the inverse sine of Value.
///
/// ## Remarks
///
/// `Est` functions offer increased performance at the expense of reduced accuracy. `Est` functions are appropriate
/// for non-critical calculations where accuracy can be sacrificed for speed. The exact amount of lost accuracy
/// and speed increase are platform dependent.
///
/// This function uses a 3-degree minimax approximation.
///
/// ## Reference
///
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarASinEst>
#[inline]
pub fn XMScalarASinEst(
    Value: f32,
) -> f32
{
    // Clamp input to [-1,1].
    let nonnegative: bool = (Value >= 0.0);
    let x: f32 = fabsf(Value);
    let mut omx: f32 = 1.0 - x;
    if (omx < 0.0)
    {
        omx = 0.0;
    }
    let root: f32 = sqrtf(omx);

    // 3-degree minimax approximation
    let mut result: f32 = ((-0.0187293 * x + 0.0742610) * x - 0.2121144) * x + 1.5707288;
    result *= root;  // acos(|x|)

    // acos(x) = pi - acos(-x) when x < 0, asin(x) = pi/2 - acos(x)
    return (if nonnegative { XM_PIDIV2 - result } else { result - XM_PIDIV2 });
}


/// Computes the arccosine of a floating-point number.
///
/// ## Parameters
///
/// `Value` float value between `-1.0` and `1.0`.
///
/// ## Return value
///
/// Returns the inverse cosine of Value.
///
/// ## Remarks
///
/// This function uses a 7-degree minimax approximation.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarACos>
#[inline]
pub fn XMScalarACos(
    Value: f32,
) -> f32
{
    // Clamp input to [-1,1].
    let nonnegative: bool = (Value >= 0.0);
    let x: f32 = fabsf(Value);
    let mut omx: f32 = 1.0 - x;
    if (omx < 0.0)
    {
        omx = 0.0;
    }
    let root: f32 = sqrtf(omx);

    // 7-degree minimax approximation
    let mut result: f32 = ((((((-0.0012624911 * x + 0.0066700901) * x - 0.0170881256) * x + 0.0308918810) * x - 0.0501743046) * x + 0.0889789874) * x - 0.2145988016) * x + 1.5707963050;
    result *= root;

    // acos(x) = pi - acos(-x) when x < 0
    return (if nonnegative { result } else { XM_PI - result });
}

/// Estimates the arccosine of a floating-point number.
///
/// ## Parameters
///
/// `Value` float value between `-1.0` and `1.0`.
///
/// ## Return value
///
/// Returns the inverse cosine of Value.
///
/// ## Remarks
///
/// `Est` functions offer increased performance at the expense of reduced accuracy. `Est` functions are appropriate
/// for non-critical calculations where accuracy can be sacrificed for speed. The exact amount of lost accuracy
/// and speed increase are platform dependent.
///
/// This function uses a 3-degree minimax approximation.
///
/// ## Reference
///
/// <https://docs.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-XMScalarACosEst>
#[inline]
pub fn XMScalarACosEst(
    Value: f32,
) -> f32
{
    // Clamp input to [-1,1].
    let nonnegative: bool = (Value >= 0.0);
    let x: f32 = fabsf(Value);
    let mut omx: f32 = 1.0 - x;
    if (omx < 0.0)
    {
        omx = 0.0;
    }
    let root: f32 = sqrtf(omx);

    // 3-degree minimax approximation
    let mut result: f32 = ((-0.0187293 * x + 0.0742610) * x - 0.2121144) * x + 1.5707288;
    result *= root;

    // acos(x) = pi - acos(-x) when x < 0
    return (if nonnegative { result } else { XM_PI - result });
}