#![allow(dead_code)]

use crate::*;

#[repr(C)]
pub enum ContainmentType {
    DISJOINT = 0,
    INTERSECTS = 1,
    CONTAINS = 2,
}

#[repr(C)]
pub enum PlaneIntersectionType {
    FRONT = 0,
    INTERSECTING = 1,
    BACK = 2,
}


/// The corners (vertices) of a triangle: `V0`, `V1`, and `V2`.
pub type Triangle = (XMVECTOR, XMVECTOR, XMVECTOR);

pub type Point = XMVECTOR;

pub type Plane = XMVECTOR;

pub type Direction = XMVECTOR;

/// `(Origin, Direction, Dist)`
///
/// `Origin` The origin of the ray.
///
/// `Direction` The direction of the ray.
///
/// `Dist` The length of the ray.
pub type Ray = (Point, Direction, f32);

#[repr(C)]
pub struct BoundingSphere {
    // Center of the sphere.
    pub Center: XMFLOAT3,

    // Radius of the sphere.
    pub Radius: f32,
}

#[repr(C)]
pub struct BoundingBox {
    // Center of the box.
    pub Center: XMFLOAT3,

    // Distance from the center to each side.
    pub Extents: XMFLOAT3,
}

impl BoundingBox {
    const CORNER_COUNT: usize = 8;
}

#[repr(C)]
pub struct BoundingOrientedBox {
    // Center of the box.
    pub Center: XMFLOAT3,

    // Distance from the center to each side.
    pub Extents: XMFLOAT3,

    // Unit quaternion representing rotation (box -> world).
    pub Orientation: XMFLOAT4,
}

impl BoundingOrientedBox {
    const CORNER_COUNT: usize = 8;
}

#[repr(C)]
pub struct BoundingFrustum {
    // Origin of the frustum (and projection).
    pub Origin: XMFLOAT3,

    // Quaternion representing rotation.
    pub Orientation: XMFLOAT4,

    // Positive X (X/Z)
    pub RightSlope: f32,

    // Negative X       
    pub LeftSlope: f32,

    // Positive Y (Y/Z)
    pub TopSlope: f32,

    // Negative Y
    pub BottomSlope: f32,

    // Z of the near plane.
    pub Near: f32,

    // Z of far plane.
    pub Far: f32,
}

impl BoundingFrustum {
    const CORNER_COUNT: usize = 8;
}

pub trait MatrixTransform: Sized {
    fn Transform(&self, Out: &mut Self, M: FXMMATRIX);
}

pub trait DecomposedTransform: Sized {
    fn Transform(&self, Out: &mut Self, Scale: f32, Rotation: FXMVECTOR, Translation: FXMVECTOR);
}

pub trait Contains<T> {
    fn Contains(&self, other: T) -> ContainmentType;
}

pub trait Intersects<T> {
    fn Intersects(&self, other: T) -> bool;
}

pub trait ContainedBy {
    fn ContainedBy(
        &self,
        Plane0: FXMVECTOR,
        Plane1: FXMVECTOR,
        Plane2: GXMVECTOR,
        Plane3: HXMVECTOR,
        Plane4: HXMVECTOR
    ) -> ContainmentType;
}

pub trait CreateMerged: Sized {
    fn CreateMerged(Out: &mut Self, B1: &Self, B2: &Self);
}

pub trait CreateFromBoundingBox: Sized {
    fn CreateFromBoundingBox(Out: &mut Self, box_: &BoundingBox);
}

pub trait CreateFromBoundingOrientedBox: Sized {
    fn CreateFromBoundingBox(Out: &mut Self, box_: &BoundingOrientedBox);
}

pub trait CreateFromPoints: Sized {
    fn CreateFromPoints<'a>(Out: &mut Self, pPoints: impl Iterator<Item=&'a XMFLOAT3>);
}

pub trait CreateFromFrustum: Sized {
    fn CreateFromFrustum(Out: &mut Self, fr: &BoundingFrustum);
}

pub trait CreateFromSphere: Sized {
    fn CreateFromSphere(Out: &mut Self, fr: &BoundingSphere);
}

const g_BoxOffset: [XMVECTORF32; 8] = [
    XMVECTORF32 { f: [ -1.0, -1.0,  1.0, 0.0 ] },
    XMVECTORF32 { f: [  1.0, -1.0,  1.0, 0.0 ] },
    XMVECTORF32 { f: [  1.0,  1.0,  1.0, 0.0 ] },
    XMVECTORF32 { f: [ -1.0,  1.0,  1.0, 0.0 ] },
    XMVECTORF32 { f: [ -1.0, -1.0, -1.0, 0.0 ] },
    XMVECTORF32 { f: [  1.0, -1.0, -1.0, 0.0 ] },
    XMVECTORF32 { f: [  1.0,  1.0, -1.0, 0.0 ] },
    XMVECTORF32 { f: [ -1.0,  1.0, -1.0, 0.0 ] },
];

const FLT_MAX: f32 = std::f32::MAX;

const g_RayEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ 1e-20, 1e-20, 1e-20, 1e-20 ] };
const g_RayNegEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ -1e-20, -1e-20, -1e-20, -1e-20 ] };
const g_FltMin: XMVECTORF32 = XMVECTORF32 { f: [ -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX ] };
const g_FltMax: XMVECTORF32 = XMVECTORF32 { f: [ FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX ] };

mod internal {
    use crate::*;

    const g_UnitVectorEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4 ] };
    const g_UnitQuaternionEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4 ] };
    const g_UnitPlaneEpsilon: XMVECTORF32 = XMVECTORF32 { f: [ 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4 ] };

    /// Return true if any of the elements of a 3 vector are equal to 0xffffffff.
    /// Slightly more efficient than using XMVector3EqualInt.
    pub fn XMVector3AnyTrue(V: FXMVECTOR) -> bool {
        // Duplicate the fourth element from the first element.
        let C: XMVECTOR = <(XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X)>::XMVectorSwizzle(V);
        return XMComparisonAnyTrue(XMVector4EqualIntR(C, XMVectorTrueInt()));
    }

    /// Return true if all of the elements of a 3 vector are equal to 0xffffffff.
    /// Slightly more efficient than using XMVector3EqualInt.
    pub fn XMVector3AllTrue(V: FXMVECTOR) -> bool {
        // Duplicate the fourth element from the first element.
        let C: XMVECTOR = <(XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_X)>::XMVectorSwizzle(V);
        return XMComparisonAllTrue(XMVector4EqualIntR(C, XMVectorTrueInt()));
    }

    /// Return true if the vector is a unit vector (length == 1).
    pub fn XMVector3IsUnit(V: FXMVECTOR) -> bool {
        let Difference: XMVECTOR = XMVectorSubtract(XMVector3Length(V), XMVectorSplatOne());
        return XMVector4Less(XMVectorAbs(Difference), g_UnitVectorEpsilon.v());
    }

    /// Return true if the quaterion is a unit quaternion.
    pub fn XMQuaternionIsUnit(Q: FXMVECTOR) -> bool {
        let Difference: XMVECTOR = XMVectorSubtract(XMVector4Length(Q), XMVectorSplatOne());
        return XMVector4Less(XMVectorAbs(Difference), g_UnitQuaternionEpsilon.v());
    }

    /// Return true if the plane is a unit plane.
    pub fn XMPlaneIsUnit(Plane: FXMVECTOR) -> bool {
        let Difference: XMVECTOR = XMVectorSubtract(XMVector3Length(Plane), XMVectorSplatOne());
        return XMVector4Less(XMVectorAbs(Difference), g_UnitPlaneEpsilon.v());
    }

    pub fn XMPlaneTransform(Plane: FXMVECTOR, Rotation: FXMVECTOR, Translation: FXMVECTOR) -> XMVECTOR {
        let vNormal: XMVECTOR = XMVector3Rotate(Plane, Rotation);
        let vD: XMVECTOR = XMVectorSubtract(XMVectorSplatW(Plane), XMVector3Dot(vNormal, Translation));

        // TODO: template
        return XMVectorInsert(vNormal, vD, 0, 0, 0, 0, 1);
    }

    /// Return the point on the line segement (S1, S2) nearest the point P.
    pub fn PointOnLineSegmentNearestPoint(S1: FXMVECTOR, S2: FXMVECTOR, P: FXMVECTOR) -> XMVECTOR {
        let Dir: XMVECTOR = XMVectorSubtract(S2, S1);
        let Projection: XMVECTOR = XMVectorSubtract(XMVector3Dot(P, Dir), XMVector3Dot(S1, Dir));
        let LengthSq: XMVECTOR = XMVector3Dot(Dir, Dir);

        let t: XMVECTOR = XMVectorMultiply(Projection, XMVectorReciprocal(LengthSq));
        let mut Point: XMVECTOR = XMVectorMultiplyAdd(t, Dir, S1);

        // t < 0
        let SelectS1: XMVECTOR = XMVectorLess(Projection, XMVectorZero());
        Point = XMVectorSelect(Point, S1, SelectS1);

        // t > 1
        let SelectS2: XMVECTOR = XMVectorGreater(Projection, LengthSq);
        Point = XMVectorSelect(Point, S2, SelectS2);

        return Point;
    }

    /// Test if the point (P) on the plane of the triangle is inside the triangle
    /// (V0, V1, V2).
    pub fn PointOnPlaneInsideTriangle(P: FXMVECTOR, V0: FXMVECTOR, V1: FXMVECTOR, V2: GXMVECTOR) -> XMVECTOR {
        // Compute the triangle normal.
        let N: XMVECTOR = XMVector3Cross(XMVectorSubtract(V2, V0), XMVectorSubtract(V1, V0));

        // Compute the cross products of the vector from the base of each edge to
        // the point with each edge vector.
        let C0: XMVECTOR = XMVector3Cross(XMVectorSubtract(P, V0), XMVectorSubtract(V1, V0));
        let C1: XMVECTOR = XMVector3Cross(XMVectorSubtract(P, V1), XMVectorSubtract(V2, V1));
        let C2: XMVECTOR = XMVector3Cross(XMVectorSubtract(P, V2), XMVectorSubtract(V0, V2));

        // If the cross product points in the same direction as the normal the the
        // point is inside the edge (it is zero if is on the edge).
        let Zero: XMVECTOR = XMVectorZero();
        let Inside0: XMVECTOR = XMVectorGreaterOrEqual(XMVector3Dot(C0, N), Zero);
        let Inside1: XMVECTOR = XMVectorGreaterOrEqual(XMVector3Dot(C1, N), Zero);
        let Inside2: XMVECTOR = XMVectorGreaterOrEqual(XMVector3Dot(C2, N), Zero);

        // If the point inside all of the edges it is inside.
        return XMVectorAndInt(XMVectorAndInt(Inside0, Inside1), Inside2);
    }

    pub fn SolveCubic(e: f32, f: f32, g: f32, t: &mut f32, u: &mut f32, v: &mut f32) -> bool {
        let p: f32;
        let q: f32;
        let h: f32;
        let rc: f32;
        let d: f32;
        let theta: f32;
        let costh3: f32;
        let sinth3: f32;

        p = f - e * e / 3.0;
        q = g - e * f / 3.0 + e * e * e * 2.0 / 27.0;
        h = q * q / 4.0 + p * p * p / 27.0;

        if (h > 0.0)
        {
            *t = 0.0;
            *u = 0.0;
            *v = 0.0;
            return false; // only one real root
        }

        if ((h == 0.0) && (q == 0.0)) // all the same root
        {
            *t = -e / 3.0;
            *u = -e / 3.0;
            *v = -e / 3.0;

            return true;
        }

        d = sqrtf(q * q / 4.0 - h);
        if (d < 0.0) {
            rc = -powf(-d, 1.0 / 3.0);
        }
        else {
            rc = powf(d, 1.0 / 3.0);
        }

        theta = XMScalarACos(-q / (2.0 * d));
        costh3 = XMScalarCos(theta / 3.0);
        sinth3 = sqrtf(3.0) * XMScalarSin(theta / 3.0);
        *t = 2.0 * rc * costh3 - e / 3.0;
        *u = -rc * (costh3 + sinth3) - e / 3.0;
        *v = -rc * (costh3 - sinth3) - e / 3.0;

        return true;
    }

    pub fn CalculateEigenVector(
        m11: f32, m12: f32, m13: f32,
        m22: f32, m23: f32, m33: f32,
        e: f32,
    ) -> XMVECTOR {
        let mut fTmp: [f32; 3] = unsafe { uninitialized() };
        fTmp[0] = m12 * m23 - m13 * (m22 - e);
        fTmp[1] = m13 * m12 - m23 * (m11 - e);
        fTmp[2] = (m11 - e) * (m22 - e) - m12 * m12;
        
        let fTmp: XMFLOAT3 = unsafe { mem::transmute(fTmp) };
        let mut vTmp: XMVECTOR = XMLoadFloat3(&fTmp);

        if (XMVector3Equal(vTmp, XMVectorZero())) // planar or linear
        {
            let f1: f32;
            let f2: f32;
            let f3: f32;

            // we only have one equation - find a valid one
            if ((m11 - e != 0.0) || (m12 != 0.0) || (m13 != 0.0))
            {
                f1 = m11 - e; f2 = m12; f3 = m13;
            }
            else if ((m12 != 0.0) || (m22 - e != 0.0) || (m23 != 0.0))
            {
                f1 = m12; f2 = m22 - e; f3 = m23;
            }
            else if ((m13 != 0.0) || (m23 != 0.0) || (m33 - e != 0.0))
            {
                f1 = m13; f2 = m23; f3 = m33 - e;
            }
            else
            {
                // error, we'll just make something up - we have NO context
                f1 = 1.0; f2 = 0.0; f3 = 0.0;
            }

            if (f1 == 0.0) {
                vTmp = XMVectorSetX(vTmp, 0.0);
            } else {
                vTmp = XMVectorSetX(vTmp, 1.0);
            }
            if (f2 == 0.0) {
                vTmp = XMVectorSetY(vTmp, 0.0);
            } else {
                vTmp = XMVectorSetY(vTmp, 1.0);
            }
            if (f3 == 0.0)
            {
                vTmp = XMVectorSetZ(vTmp, 0.0);
                // recalculate y to make equation work
                if (m12 != 0.0) {
                    vTmp = XMVectorSetY(vTmp, -f1 / f2);
                }
            }
            else
            {
                vTmp = XMVectorSetZ(vTmp, (f2 - f1) / f3);
            }
        }

        if (XMVectorGetX(XMVector3LengthSq(vTmp)) > 1e-5)
        {
            return XMVector3Normalize(vTmp);
        }
        else
        {
            // Multiply by a value large enough to make the vector non-zero.
            vTmp = XMVectorScale(vTmp, 1e5);
            return XMVector3Normalize(vTmp);
        }
    }

    pub fn CalculateEigenVectors(
        m11: f32, m12: f32, m13: f32,
        m22: f32, m23: f32, m33: f32,
        e1: f32, e2: f32, e3: f32,
        pV1: &mut XMVECTOR,
        pV2: &mut XMVECTOR,
        pV3: &mut XMVECTOR,
    ) -> bool {
        *pV1 = CalculateEigenVector(m11, m12, m13, m22, m23, m33, e1);
        *pV2 = CalculateEigenVector(m11, m12, m13, m22, m23, m33, e2);
        *pV3 = CalculateEigenVector(m11, m12, m13, m22, m23, m33, e3);

        let mut v1z: bool = false;
        let mut v2z: bool = false;
        let mut v3z: bool = false;

        let Zero: XMVECTOR = XMVectorZero();

        if (XMVector3Equal(*pV1, Zero)) {
            v1z = true;
        }

        if (XMVector3Equal(*pV2, Zero)) {
            v2z = true;
        }

        if (XMVector3Equal(*pV3, Zero)) {
            v3z = true;
        }

        let e12: bool = (fabsf(XMVectorGetX(XMVector3Dot(*pV1, *pV2))) > 0.1); // check for non-orthogonal vectors
        let e13: bool = (fabsf(XMVectorGetX(XMVector3Dot(*pV1, *pV3))) > 0.1);
        let e23: bool = (fabsf(XMVectorGetX(XMVector3Dot(*pV2, *pV3))) > 0.1);

        if ((v1z && v2z && v3z) || (e12 && e13 && e23) ||
            (e12 && v3z) || (e13 && v2z) || (e23 && v1z)) // all eigenvectors are 0- any basis set
        {
            *pV1 = g_XMIdentityR0.v();
            *pV2 = g_XMIdentityR1.v();
            *pV3 = g_XMIdentityR2.v();
            return true;
        }

        if (v1z && v2z)
        {
            let mut vTmp: XMVECTOR = XMVector3Cross(g_XMIdentityR1.v(), *pV3);
            if (XMVectorGetX(XMVector3LengthSq(vTmp)) < 1e-5)
            {
                vTmp = XMVector3Cross(g_XMIdentityR0.v(), *pV3);
            }
            *pV1 = XMVector3Normalize(vTmp);
            *pV2 = XMVector3Cross(*pV3, *pV1);
            return true;
        }

        if (v3z && v1z)
        {
            let mut vTmp: XMVECTOR = XMVector3Cross(g_XMIdentityR1.v(), *pV2);
            if (XMVectorGetX(XMVector3LengthSq(vTmp)) < 1e-5)
            {
                vTmp = XMVector3Cross(g_XMIdentityR0.v(), *pV2);
            }
            *pV3 = XMVector3Normalize(vTmp);
            *pV1 = XMVector3Cross(*pV2, *pV3);
            return true;
        }

        if (v2z && v3z)
        {
            let mut vTmp: XMVECTOR = XMVector3Cross(g_XMIdentityR1.v(), *pV1);
            if (XMVectorGetX(XMVector3LengthSq(vTmp)) < 1e-5)
            {
                vTmp = XMVector3Cross(g_XMIdentityR0.v(), *pV1);
            }
            *pV2 = XMVector3Normalize(vTmp);
            *pV3 = XMVector3Cross(*pV1, *pV2);
            return true;
        }

        if ((v1z) || e12)
        {
            *pV1 = XMVector3Cross(*pV2, *pV3);
            return true;
        }

        if ((v2z) || e23)
        {
            *pV2 = XMVector3Cross(*pV3, *pV1);
            return true;
        }

        if ((v3z) || e13)
        {
            *pV3 = XMVector3Cross(*pV1, *pV2);
            return true;
        }

        return true;
    }

    pub fn CalculateEigenVectorsFromCovarianceMatrix(
        Cxx: f32,
        Cyy: f32,
        Czz: f32,
        Cxy: f32,
        Cxz: f32,
        Cyz: f32,
        pV1: &mut XMVECTOR,
        pV2: &mut XMVECTOR,
        pV3: &mut XMVECTOR,

    ) -> bool {
        // Calculate the eigenvalues by solving a cubic equation.
        let e: f32 = -(Cxx + Cyy + Czz);
        let f: f32 = Cxx * Cyy + Cyy * Czz + Czz * Cxx - Cxy * Cxy - Cxz * Cxz - Cyz * Cyz;
        let g: f32 = Cxy * Cxy * Czz + Cxz * Cxz * Cyy + Cyz * Cyz * Cxx - Cxy * Cyz * Cxz * 2.0 - Cxx * Cyy * Czz;

        let mut ev1: f32 = 0.0;
        let mut ev2: f32 = 0.0;
        let mut ev3: f32 = 0.0;
        if (!SolveCubic(e, f, g, &mut ev1, &mut ev2, &mut ev3))
        {
            // set them to arbitrary orthonormal basis set
            *pV1 = g_XMIdentityR0.v();
            *pV2 = g_XMIdentityR1.v();
            *pV3 = g_XMIdentityR2.v();
            return false;
        }

        return CalculateEigenVectors(Cxx, Cxy, Cxz, Cyy, Cyz, Czz, ev1, ev2, ev3, pV1, pV2, pV3);
    }

    pub fn FastIntersectTrianglePlane(
        V0: FXMVECTOR,
        V1: FXMVECTOR,
        V2: FXMVECTOR,
        Plane: GXMVECTOR,
        Outside: &mut XMVECTOR,
        Inside: &mut XMVECTOR
    ) {
        // Plane0
        let Dist0: XMVECTOR = XMVector4Dot(V0, Plane);
        let Dist1: XMVECTOR = XMVector4Dot(V1, Plane);
        let Dist2: XMVECTOR = XMVector4Dot(V2, Plane);

        let mut MinDist: XMVECTOR = XMVectorMin(Dist0, Dist1);
        MinDist = XMVectorMin(MinDist, Dist2);

        let mut MaxDist: XMVECTOR = XMVectorMax(Dist0, Dist1);
        MaxDist = XMVectorMax(MaxDist, Dist2);

        let Zero: XMVECTOR = XMVectorZero();

        // Outside the plane?
        *Outside = XMVectorGreater(MinDist, Zero);

        // Fully inside the plane?
        *Inside = XMVectorLess(MaxDist, Zero);
    }

    pub fn FastIntersectSpherePlane(
        Center: FXMVECTOR,
        Radius: FXMVECTOR,
        Plane: FXMVECTOR,
        Outside: &mut XMVECTOR,
        Inside: &mut XMVECTOR,
    ) {
        let Dist: XMVECTOR = XMVector4Dot(Center, Plane);

        // Outside the plane?
        *Outside = XMVectorGreater(Dist, Radius);

        // Fully inside the plane?
        *Inside = XMVectorLess(Dist, XMVectorNegate(Radius));
    }

    pub fn FastIntersectAxisAlignedBoxPlane(
        Center: FXMVECTOR,
        Extents: FXMVECTOR,
        Plane: FXMVECTOR,
        Outside: &mut XMVECTOR,
        Inside: &mut XMVECTOR,
    ) {
         // Compute the distance to the center of the box.
         let Dist: XMVECTOR = XMVector4Dot(Center, Plane);

         // Project the axes of the box onto the normal of the plane.  Half the
         // length of the projection (sometime called the "radius") is equal to
         // h(u) * abs(n dot b(u))) + h(v) * abs(n dot b(v)) + h(w) * abs(n dot b(w))
         // where h(i) are extents of the box, n is the plane normal, and b(i) are the
         // axes of the box. In this case b(i) = [(1,0,0), (0,1,0), (0,0,1)].
         let Radius: XMVECTOR = XMVector3Dot(Extents, XMVectorAbs(Plane));
 
         // Outside the plane?
         *Outside = XMVectorGreater(Dist, Radius);
 
         // Fully inside the plane?
         *Inside = XMVectorLess(Dist, XMVectorNegate(Radius));
    }

    pub fn FastIntersectOrientedBoxPlane(
        Center: FXMVECTOR,
        Extents: FXMVECTOR,
        Axis0: FXMVECTOR,
        Axis1: GXMVECTOR,
        Axis2: HXMVECTOR,
        Plane: HXMVECTOR,
        Outside: &mut XMVECTOR,
        Inside: &mut XMVECTOR,
    ) {
        // Compute the distance to the center of the box.
        let Dist: XMVECTOR = XMVector4Dot(Center, Plane);

        // Project the axes of the box onto the normal of the plane.  Half the
        // length of the projection (sometime called the "radius") is equal to
        // h(u) * abs(n dot b(u))) + h(v) * abs(n dot b(v)) + h(w) * abs(n dot b(w))
        // where h(i) are extents of the box, n is the plane normal, and b(i) are the
        // axes of the box.
        let mut Radius: XMVECTOR = XMVector3Dot(Plane, Axis0);
        // TODO: template
        Radius = XMVectorInsert(Radius, XMVector3Dot(Plane, Axis1), 0, 0, 1, 0, 0);
        Radius = XMVectorInsert(Radius, XMVector3Dot(Plane, Axis2), 0, 0, 0, 1, 0);
        Radius = XMVector3Dot(Extents, XMVectorAbs(Radius));

        // Outside the plane?
        *Outside = XMVectorGreater(Dist, Radius);

        // Fully inside the plane?
        *Inside = XMVectorLess(Dist, XMVectorNegate(Radius));
    }

    pub fn FastIntersectFrustumPlane(
        Point0: FXMVECTOR,
        Point1: FXMVECTOR,
        Point2: FXMVECTOR,
        Point3: GXMVECTOR,
        Point4: HXMVECTOR,
        Point5: HXMVECTOR,
        Point6: CXMVECTOR,
        Point7: CXMVECTOR,
        Plane: CXMVECTOR,
        Outside: &mut XMVECTOR,
        Inside: &mut XMVECTOR,
    ) {
        let Plane = *Plane;
        let Point6 = *Point6;
        let Point7 = *Point7;

        // Find the min/max projection of the frustum onto the plane normal.
        let mut Min: XMVECTOR = XMVector3Dot(Plane, Point0);
        let mut Max: XMVECTOR = Min;
        let mut Dist: XMVECTOR;

        Dist = XMVector3Dot(Plane, Point1);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        Dist = XMVector3Dot(Plane, Point2);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        Dist = XMVector3Dot(Plane, Point3);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        Dist = XMVector3Dot(Plane, Point4);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        Dist = XMVector3Dot(Plane, Point5);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        Dist = XMVector3Dot(Plane, Point6);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        Dist = XMVector3Dot(Plane, Point7);
        Min = XMVectorMin(Min, Dist);
        Max = XMVectorMax(Max, Dist);

        let PlaneDist: XMVECTOR = XMVectorNegate(XMVectorSplatW(Plane));

        // Outside the plane?
        *Outside = XMVectorGreater(Min, PlaneDist);

        // Fully inside the plane?
        *Inside = XMVectorLess(Max, PlaneDist);
    }
}

// BoundingSphere -------------------------------------------------------------

impl MatrixTransform for BoundingSphere {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-transform>
    fn Transform(&self, Out: &mut Self, M: FXMMATRIX) {
        unsafe {
            // Load the center of the sphere.
            let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);

            // Transform the center of the sphere.
            let C: XMVECTOR = XMVector3Transform(vCenter, M);

            let dX: XMVECTOR = XMVector3Dot(M.r[0], M.r[0]);
            let dY: XMVECTOR = XMVector3Dot(M.r[1], M.r[1]);
            let dZ: XMVECTOR = XMVector3Dot(M.r[2], M.r[2]);

            let d: XMVECTOR = XMVectorMax(dX, XMVectorMax(dY, dZ));

            // Store the center sphere.
            XMStoreFloat3(&mut Out.Center, C);

            // Scale the radius of the pshere.
            let Scale: f32 = sqrtf(XMVectorGetX(d));
            Out.Radius = self.Radius * Scale;
        }
    }
}

impl DecomposedTransform for BoundingSphere {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-transform(boundingsphere__float_fxmvector_fxmvector)>
    fn Transform(&self, Out: &mut Self, Scale: f32, Rotation: FXMVECTOR, Translation: FXMVECTOR) {
        // Load the center of the sphere.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);

        // Transform the center of the sphere.
        vCenter = XMVectorAdd(XMVector3Rotate(XMVectorScale(vCenter, Scale), Rotation), Translation);

        // Store the center sphere.
        XMStoreFloat3(&mut Out.Center, vCenter);

        // Scale the radius of the pshere.
        Out.Radius = self.Radius * Scale;
    }
}

impl Contains<Point> for BoundingSphere {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains>
    fn Contains(&self, Point: Point) -> ContainmentType {
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);

        let DistanceSquared: XMVECTOR = XMVector3LengthSq(XMVectorSubtract(Point, vCenter));
        let RadiusSquared: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        if XMVector3LessOrEqual(DistanceSquared, RadiusSquared) {
            ContainmentType::CONTAINS
        } else {
            ContainmentType::DISJOINT
        }
    }
}

impl Contains<Triangle> for BoundingSphere {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (V0, V1, V2): Triangle) -> ContainmentType {
        if (!self.Intersects((V0, V1, V2))) {
            return ContainmentType::DISJOINT;
        }

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);
        let RadiusSquared: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        let mut DistanceSquared: XMVECTOR = XMVector3LengthSq(XMVectorSubtract(V0, vCenter));
        let mut Inside: XMVECTOR = XMVectorLessOrEqual(DistanceSquared, RadiusSquared);

        DistanceSquared = XMVector3LengthSq(XMVectorSubtract(V1, vCenter));
        Inside = XMVectorAndInt(Inside, XMVectorLessOrEqual(DistanceSquared, RadiusSquared));

        DistanceSquared = XMVector3LengthSq(XMVectorSubtract(V2, vCenter));
        Inside = XMVectorAndInt(Inside, XMVectorLessOrEqual(DistanceSquared, RadiusSquared));

        if (XMVector3EqualInt(Inside, XMVectorTrueInt())) {
            ContainmentType::CONTAINS
        } else {
            ContainmentType::INTERSECTS
        }
    }
}

impl Contains<&BoundingSphere> for BoundingSphere {
    /// Tests whether the BoundingSphere contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingsphere_)>
    fn Contains(&self, _sh: &BoundingSphere) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingBox> for BoundingSphere {
    /// Tests whether the BoundingSphere contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingbox_)>
    fn Contains(&self, _box: &BoundingBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingOrientedBox> for BoundingSphere {
    /// Tests whether the BoundingSphere contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingorientedbox_)>
    fn Contains(&self, _box: &BoundingOrientedBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingFrustum> for BoundingSphere {
    /// Tests whether the BoundingSphere contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingfrustum_)>
    fn Contains(&self, _fr: &BoundingFrustum) -> ContainmentType {
        todo!()
    }
}

impl Intersects<&BoundingSphere> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects>
    fn Intersects(&self, _sh: &BoundingSphere) -> bool { todo!() }
}

impl Intersects<&BoundingBox> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(constboundingbox_)>
    fn Intersects(&self, _box: &BoundingBox) -> bool { todo!() }
}

impl Intersects<&BoundingOrientedBox> for BoundingSphere {
    /// Test the BoundingSphere for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(constboundingorientedbox_)>
    fn Intersects(&self, _box: &BoundingOrientedBox) -> bool { todo!() }
}

impl Intersects<&BoundingFrustum> for BoundingSphere {
    /// Test the BoundingSphere for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(constboundingfrustum_)>
    fn Intersects(&self, _box: &BoundingFrustum) -> bool { todo!() }
}

impl Intersects<Triangle> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (_V0, _V1, _V2): Triangle) -> bool { todo!() }
}

impl Intersects<Plane> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(fxmvector)>
    fn Intersects(&self, _Plane: Plane) -> bool { todo!() }
}

impl Intersects<Ray> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (_Origin, _Direction, _Dist): Ray) -> bool { todo!() }
}

impl ContainedBy for BoundingSphere {
    /// Tests whether the BoundingSphere is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-containedby>
    fn ContainedBy(
        &self,
        _Plane0: FXMVECTOR,
        _Plane1: FXMVECTOR,
        _Plane2: GXMVECTOR,
        _Plane3: HXMVECTOR,
        _Plane4: HXMVECTOR
    ) -> ContainmentType
    {
        todo!()
    }
}

impl CreateMerged for BoundingSphere {
    /// Creates a BoundingSphere that contains the two specified BoundingSphere objects.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createmerged>
    fn CreateMerged(_Out: &mut Self, _B1: &Self, _B2: &Self) {
        todo!()
    }
}


impl CreateFromBoundingBox for BoundingSphere {
    /// Creates a BoundingSphere containing the specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createfromboundingbox>
    fn CreateFromBoundingBox(_Out: &mut Self, _box_: &BoundingBox) { todo!() }
}

impl CreateFromBoundingOrientedBox for BoundingSphere {
    /// Creates a BoundingSphere containing the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createfromboundingbox(boundingsphere__constboundingorientedbox_)>
    fn CreateFromBoundingBox(_Out: &mut Self, _box_: &BoundingOrientedBox) { todo!() }
}

impl CreateFromPoints for BoundingSphere {
    /// Creates a new BoundingSphere from a list of points.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createfrompoints>
    fn CreateFromPoints<'a>(_Out: &mut Self, _pPoints: impl Iterator<Item=&'a XMFLOAT3>) { todo!() }
}

impl CreateFromFrustum for BoundingSphere {
    /// Creates a BoundingSphere containing the specified BoundingFrustum.
    fn CreateFromFrustum(_Out: &mut Self, _fr: &BoundingFrustum) { todo!() }
}


// BoundingBox ----------------------------------------------------------------

impl BoundingBox {
    pub fn GetCorners(&self, _Corners: &mut [XMFLOAT3; 8]) {
        todo!()
    }
}

impl MatrixTransform for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-transform>
    fn Transform(&self, _Out: &mut Self, _M: FXMMATRIX) {
        todo!()
    }
}

impl DecomposedTransform for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-transform(BoundingBox__float_fxmvector_fxmvector)>
    fn Transform(&self, _Out: &mut Self, _Scale: f32, _Rotation: FXMVECTOR, _Translation: FXMVECTOR) {
        todo!()
    }
}

impl Contains<Point> for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains>
    fn Contains(&self, _Point: Point) -> ContainmentType {
        todo!()
    }
}

impl Contains<Triangle> for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (_V0, _V1, _V2): Triangle) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingSphere> for BoundingBox {
    /// Tests whether the BoundingBox contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingsphere_)>
    fn Contains(&self, _sh: &BoundingSphere) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingBox> for BoundingBox {
    /// Tests whether the BoundingBox contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingbox_)>
    fn Contains(&self, _box: &BoundingBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingOrientedBox> for BoundingBox {
    /// Tests whether the BoundingBox contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingorientedbox_)>
    fn Contains(&self, _box: &BoundingOrientedBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingFrustum> for BoundingBox {
    /// Tests whether the BoundingBox contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingfrustum_)>
    fn Contains(&self, _fr: &BoundingFrustum) -> ContainmentType {
        todo!()
    }
}

impl Intersects<&BoundingSphere> for BoundingBox {
    /// Tests the BoundingBox for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects>
    fn Intersects(&self, _sh: &BoundingSphere) -> bool { todo!() }
}

impl Intersects<&BoundingBox> for BoundingBox {
    /// Tests the BoundingBox for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(constboundingbox_)>
    fn Intersects(&self, _box: &BoundingBox) -> bool { todo!() }
}

impl Intersects<&BoundingOrientedBox> for BoundingBox {
    /// Test the BoundingBox for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(constboundingorientedbox_)>
    fn Intersects(&self, _box: &BoundingOrientedBox) -> bool { todo!() }
}

impl Intersects<&BoundingFrustum> for BoundingBox {
    /// Test the BoundingBox for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(constboundingfrustum_)>
    fn Intersects(&self, _box: &BoundingFrustum) -> bool { todo!() }
}

impl Intersects<Triangle> for BoundingBox {
    /// Tests the BoundingSphere for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (_V0, _V1, _V2): Triangle) -> bool { todo!() }
}

impl Intersects<Plane> for BoundingBox {
    /// Tests the BoundingBox for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(fxmvector)>
    fn Intersects(&self, _Plane: Plane) -> bool { todo!() }
}

impl Intersects<Ray> for BoundingBox {
    /// Tests the BoundingBox for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (_Origin, _Direction, _Dist): Ray) -> bool { todo!() }
}

impl ContainedBy for BoundingBox {
    /// Tests whether the BoundingBox is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-containedby>
    fn ContainedBy(
        &self,
        _Plane0: FXMVECTOR,
        _Plane1: FXMVECTOR,
        _Plane2: GXMVECTOR,
        _Plane3: HXMVECTOR,
        _Plane4: HXMVECTOR
    ) -> ContainmentType
    {
        todo!()
    }
}

impl CreateMerged for BoundingBox {
    /// Creates a BoundingBox that contains the two specified BoundingBox objects.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-createmerged>
    fn CreateMerged(_Out: &mut Self, _B1: &Self, _B2: &Self) {
        todo!()
    }
}

impl CreateFromSphere for BoundingBox {
    /// Creates a BoundingBox large enough to contain the a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingbox-createfromsphere>
    fn CreateFromSphere(_Out: &mut Self, _box: &BoundingSphere) { todo!() }
}

impl CreateFromPoints for BoundingBox {
    /// Creates a new BoundingBox from a list of points.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-createfrompoints>
    fn CreateFromPoints<'a>(_Out: &mut Self, _pPoints: impl Iterator<Item=&'a XMFLOAT3>) { todo!() }
}

// BoundingOrientedBox --------------------------------------------------------

impl MatrixTransform for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-transform>
    fn Transform(&self, _Out: &mut Self, _M: FXMMATRIX) {
        todo!()
    }
}

impl DecomposedTransform for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-transform(BoundingOrientedBox__float_fxmvector_fxmvector)>
    fn Transform(&self, _Out: &mut Self, _Scale: f32, _Rotation: FXMVECTOR, _Translation: FXMVECTOR) {
        todo!()
    }
}

impl Contains<Point> for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains>
    fn Contains(&self, _Point: Point) -> ContainmentType {
        todo!()
    }
}

impl Contains<Triangle> for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (_V0, _V1, _V2): Triangle) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingSphere> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingsphere_)>
    fn Contains(&self, _sh: &BoundingSphere) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingBox> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingbox_)>
    fn Contains(&self, _box: &BoundingBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingOrientedBox> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingorientedbox_)>
    fn Contains(&self, _box: &BoundingOrientedBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingFrustum> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingfrustum_)>
    fn Contains(&self, _fr: &BoundingFrustum) -> ContainmentType {
        todo!()
    }
}

impl Intersects<&BoundingSphere> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects>
    fn Intersects(&self, _sh: &BoundingSphere) -> bool { todo!() }
}

impl Intersects<&BoundingBox> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(constboundingbox_)>
    fn Intersects(&self, _box: &BoundingBox) -> bool { todo!() }
}

impl Intersects<&BoundingOrientedBox> for BoundingOrientedBox {
    /// Test the BoundingOrientedBox for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(constboundingorientedbox_)>
    fn Intersects(&self, _box: &BoundingOrientedBox) -> bool { todo!() }
}

impl Intersects<&BoundingFrustum> for BoundingOrientedBox {
    /// Test the BoundingOrientedBox for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(constboundingfrustum_)>
    fn Intersects(&self, _box: &BoundingFrustum) -> bool { todo!() }
}

impl Intersects<Triangle> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (_V0, _V1, _V2): Triangle) -> bool { todo!() }
}

impl Intersects<Plane> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(fxmvector)>
    fn Intersects(&self, _Plane: Plane) -> bool { todo!() }
}

impl Intersects<Ray> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (_Origin, _Direction, _Dist): Ray) -> bool { todo!() }
}

impl ContainedBy for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-containedby>
    fn ContainedBy(
        &self,
        _Plane0: FXMVECTOR,
        _Plane1: FXMVECTOR,
        _Plane2: GXMVECTOR,
        _Plane3: HXMVECTOR,
        _Plane4: HXMVECTOR
    ) -> ContainmentType
    {
        todo!()
    }
}

impl CreateFromBoundingBox for BoundingOrientedBox {
    /// Creates a BoundingBox large enough to contain the a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-CreateFromBoundingBox>
    fn CreateFromBoundingBox(_Out: &mut Self, _box: &BoundingBox) { todo!() }
}

impl CreateFromPoints for BoundingOrientedBox {
    /// Creates a new BoundingOrientedBox from a list of points.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-createfrompoints>
    fn CreateFromPoints<'a>(_Out: &mut Self, _pPoints: impl Iterator<Item=&'a XMFLOAT3>) { todo!() }
}

// BoundingFrustum ----------------------------------------------------------------

impl MatrixTransform for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-transform>
    fn Transform(&self, _Out: &mut Self, _M: FXMMATRIX) {
        todo!()
    }
}

impl DecomposedTransform for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-transform(BoundingFrustum__float_fxmvector_fxmvector)>
    fn Transform(&self, _Out: &mut Self, _Scale: f32, _Rotation: FXMVECTOR, _Translation: FXMVECTOR) {
        todo!()
    }
}

impl Contains<Point> for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains>
    fn Contains(&self, _Point: Point) -> ContainmentType {
        todo!()
    }
}

impl Contains<Triangle> for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (_V0, _V1, _V2): Triangle) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingSphere> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingsphere_)>
    fn Contains(&self, _sh: &BoundingSphere) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingBox> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingbox_)>
    fn Contains(&self, _box: &BoundingBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingOrientedBox> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingorientedbox_)>
    fn Contains(&self, _box: &BoundingOrientedBox) -> ContainmentType {
        todo!()
    }
}

impl Contains<&BoundingFrustum> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingfrustum_)>
    fn Contains(&self, _fr: &BoundingFrustum) -> ContainmentType {
        todo!()
    }
}

impl Intersects<&BoundingSphere> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects>
    fn Intersects(&self, _sh: &BoundingSphere) -> bool { todo!() }
}

impl Intersects<&BoundingBox> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(constboundingbox_)>
    fn Intersects(&self, _box: &BoundingBox) -> bool { todo!() }
}

impl Intersects<&BoundingOrientedBox> for BoundingFrustum {
    /// Test the BoundingFrustum for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(constboundingorientedbox_)>
    fn Intersects(&self, _box: &BoundingOrientedBox) -> bool { todo!() }
}

impl Intersects<&BoundingFrustum> for BoundingFrustum {
    /// Test the BoundingFrustum for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(constboundingfrustum_)>
    fn Intersects(&self, _box: &BoundingFrustum) -> bool { todo!() }
}

impl Intersects<Triangle> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (_V0, _V1, _V2): Triangle) -> bool { todo!() }
}

impl Intersects<Plane> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(fxmvector)>
    fn Intersects(&self, _Plane: Plane) -> bool { todo!() }
}

impl Intersects<Ray> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (_Origin, _Direction, _Dist): Ray) -> bool { todo!() }
}

impl ContainedBy for BoundingFrustum {
    /// Tests whether the BoundingFrustum is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-containedby>
    fn ContainedBy(
        &self,
        _Plane0: FXMVECTOR,
        _Plane1: FXMVECTOR,
        _Plane2: GXMVECTOR,
        _Plane3: HXMVECTOR,
        _Plane4: HXMVECTOR
    ) -> ContainmentType
    {
        todo!()
    }
}

impl BoundingFrustum {
    pub fn CreateFromMatrix(_Out: &mut Self, _Projection: FXMMATRIX) {
        todo!()
    }
    pub fn GetPlanes(
        _NearPlane: Option<&mut XMVECTOR>,
        _FarPlane: Option<&mut XMVECTOR>,
        _RightPlane: Option<&mut XMVECTOR>,
        _LeftPlane: Option<&mut XMVECTOR>,
        _TopPlane: Option<&mut XMVECTOR>,
        _BottomPlane: Option<&mut XMVECTOR>,
    ) {
        todo!()
    }
}

pub mod triangle_tests {
    use crate::*;
    pub fn IntersectsRay(_Origin: FXMVECTOR, _Direction: FXMVECTOR, _V0: FXMVECTOR, _V1: GXMVECTOR, _V2: HXMVECTOR, _Dist: &f32) {
        todo!()
    }

    pub fn IntersectsTriangle(
        _A0: FXMVECTOR,
        _A1: FXMVECTOR,
        _A2: FXMVECTOR,
        _B0: GXMVECTOR,
        _B1: HXMVECTOR,
        _B2: HXMVECTOR,
      ) -> bool {
          todo!()
      }

      pub fn IntersectsPlane(
        _V0: FXMVECTOR,
        _V1: FXMVECTOR,
        _V2: FXMVECTOR,
        _Plane: GXMVECTOR,
      ) -> bool {
          todo!()
      }
}