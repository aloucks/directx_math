#![allow(dead_code)]

use crate::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContainmentType {
    DISJOINT = 0,
    INTERSECTS = 1,
    CONTAINS = 2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PlaneIntersectionType {
    FRONT = 0,
    INTERSECTING = 1,
    BACK = 2,
}

use ContainmentType::{DISJOINT, INTERSECTS, CONTAINS};
use PlaneIntersectionType::{FRONT, INTERSECTING, BACK};

/// The corners (vertices) of a triangle: `V0`, `V1`, and `V2`.
pub type Triangle = (XMVECTOR, XMVECTOR, XMVECTOR);

/// 3D Vector
pub type Point = XMVECTOR;

pub type Plane = XMVECTOR;

/// Unit 3D Vector
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
#[derive(Copy, Clone, Debug)]
pub struct BoundingSphere {
    // Center of the sphere.
    pub Center: XMFLOAT3,

    // Radius of the sphere.
    pub Radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
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
#[derive(Copy, Clone, Debug)]
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

pub trait Intersects<T, U = bool> {
    fn Intersects(&self, other: T) -> U;
}

pub trait ContainedBy {
    fn ContainedBy(
        &self,
        Plane0: FXMVECTOR,
        Plane1: FXMVECTOR,
        Plane2: GXMVECTOR,
        Plane3: HXMVECTOR,
        Plane4: HXMVECTOR,
        Plane5: HXMVECTOR,
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
    fn CreateFromPoints<'a>(Out: &mut Self, pPoints: impl Iterator<Item=&'a XMFLOAT3> + Clone);
}

pub trait CreateFromFrustum: Sized {
    fn CreateFromFrustum(Out: &mut Self, fr: &BoundingFrustum);
}

pub trait CreateFromSphere: Sized {
    fn CreateFromSphere(Out: &mut Self, fr: &BoundingSphere);
}

pub trait CreateFromMatrix: Sized {
    fn CreateFromMatrix(Out: &mut Self, Projection: FXMMATRIX);
}

const g_BoxOffset: [XMVECTORF32; BoundingBox::CORNER_COUNT] = [
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

        return if (XMVector3EqualInt(Inside, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS }
    }
}

impl Contains<&BoundingSphere> for BoundingSphere {
    /// Tests whether the BoundingSphere contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingsphere_)>
    fn Contains(&self, sh: &BoundingSphere) -> ContainmentType {
        let Center1: XMVECTOR = XMLoadFloat3(&self.Center);
        let r1: f32 = self.Radius;

        let Center2: XMVECTOR = XMLoadFloat3(&sh.Center);
        let r2: f32 = sh.Radius;

        let V: XMVECTOR = XMVectorSubtract(Center2, Center1);

        let Dist: XMVECTOR = XMVector3Length(V);

        let d: f32 = XMVectorGetX(Dist);
        
        return if (r1 + r2 >= d) { if (r1 - r2 >= d) { CONTAINS } else { INTERSECTS } } else { DISJOINT }
    }
}

impl Contains<&BoundingBox> for BoundingSphere {
    /// Tests whether the BoundingSphere contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingbox_)>
    fn Contains(&self, box_: &BoundingBox) -> ContainmentType {
        if (!box_.Intersects(self)) {
            return DISJOINT;
        }

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);
        let RadiusSq: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        let boxCenter: XMVECTOR = XMLoadFloat3(&box_.Center);
        let boxExtents: XMVECTOR = XMLoadFloat3(&box_.Extents);

        let mut InsideAll: XMVECTOR = XMVectorTrueInt();

        let offset: XMVECTOR = XMVectorSubtract(boxCenter, vCenter);

        for i in 0 .. BoundingBox::CORNER_COUNT {
            let C: XMVECTOR = XMVectorMultiplyAdd(boxExtents, g_BoxOffset[i].v(), offset);
            let d: XMVECTOR = XMVector3LengthSq(C);
            InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(d, RadiusSq));
        }

        return if (XMVector3EqualInt(InsideAll, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingOrientedBox> for BoundingSphere {
    /// Tests whether the BoundingSphere contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingorientedbox_)>
    fn Contains(&self, box_: &BoundingOrientedBox) -> ContainmentType {
        if (!box_.Intersects(self)) {
            return DISJOINT;
        }

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);
        let RadiusSq: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        let boxCenter: XMVECTOR = XMLoadFloat3(&box_.Center);
        let boxExtents: XMVECTOR = XMLoadFloat3(&box_.Extents);
        let boxOrientation: XMVECTOR = XMLoadFloat4(&box_.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(boxOrientation));

        let mut InsideAll: XMVECTOR = XMVectorTrueInt();

        for i in 0 .. BoundingOrientedBox::CORNER_COUNT {
            let C: XMVECTOR = XMVectorAdd(XMVector3Rotate(XMVectorMultiply(boxExtents, g_BoxOffset[i].v()), boxOrientation), boxCenter);
            let d: XMVECTOR = XMVector3LengthSq(XMVectorSubtract(vCenter, C));
            InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(d, RadiusSq));
        }

        return if (XMVector3EqualInt(InsideAll, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingFrustum> for BoundingSphere {
    /// Tests whether the BoundingSphere contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-contains(constboundingfrustum_)>
    fn Contains(&self, fr: &BoundingFrustum) -> ContainmentType {
        if (!fr.Intersects(self)) {
            return DISJOINT;
        }

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);
        let RadiusSq: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        let vOrigin: XMVECTOR = XMLoadFloat3(&fr.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&fr.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        // Build the corners of the frustum.
        let vRightTop: XMVECTOR = XMVectorSet(fr.RightSlope, fr.TopSlope, 1.0, 0.0);
        let vRightBottom: XMVECTOR = XMVectorSet(fr.RightSlope, fr.BottomSlope, 1.0, 0.0);
        let vLeftTop: XMVECTOR = XMVectorSet(fr.LeftSlope, fr.TopSlope, 1.0, 0.0);
        let vLeftBottom: XMVECTOR = XMVectorSet(fr.LeftSlope, fr.BottomSlope, 1.0, 0.0);
        let vNear: XMVECTOR = XMVectorReplicatePtr(&fr.Near);
        let vFar: XMVECTOR = XMVectorReplicatePtr(&fr.Far);

        let mut Corners: [XMVECTOR; BoundingFrustum::CORNER_COUNT] = unsafe { uninitialized() };
        Corners[0] = XMVectorMultiply(vRightTop, vNear);
        Corners[1] = XMVectorMultiply(vRightBottom, vNear);
        Corners[2] = XMVectorMultiply(vLeftTop, vNear);
        Corners[3] = XMVectorMultiply(vLeftBottom, vNear);
        Corners[4] = XMVectorMultiply(vRightTop, vFar);
        Corners[5] = XMVectorMultiply(vRightBottom, vFar);
        Corners[6] = XMVectorMultiply(vLeftTop, vFar);
        Corners[7] = XMVectorMultiply(vLeftBottom, vFar);

        let mut InsideAll: XMVECTOR = XMVectorTrueInt();
        for i in 0..BoundingFrustum::CORNER_COUNT {
            let C: XMVECTOR = XMVectorAdd(XMVector3Rotate(Corners[i], vOrientation), vOrigin);
            let d: XMVECTOR = XMVector3LengthSq(XMVectorSubtract(vCenter, C));
            InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(d, RadiusSq));
        }

        return if (XMVector3EqualInt(InsideAll, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Intersects<&BoundingSphere> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects>
    fn Intersects(&self, sh: &BoundingSphere) -> bool { 
        // Load A.
        let vCenterA: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadiusA: XMVECTOR = XMVectorReplicatePtr(&self.Radius);

        // Load B.
        let vCenterB: XMVECTOR = XMLoadFloat3(&sh.Center);
        let vRadiusB: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);

        // Distance squared between centers.
        let Delta: XMVECTOR = XMVectorSubtract(vCenterB, vCenterA);
        let DistanceSquared: XMVECTOR = XMVector3LengthSq(Delta);

        // Sum of the radii squared.
        let mut RadiusSquared: XMVECTOR = XMVectorAdd(vRadiusA, vRadiusB);
        RadiusSquared = XMVectorMultiply(RadiusSquared, RadiusSquared);

        return XMVector3LessOrEqual(DistanceSquared, RadiusSquared);
    }
}

impl Intersects<&BoundingBox> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(constboundingbox_)>
    fn Intersects(&self, box_: &BoundingBox) -> bool { 
        return box_.Intersects(self);
    }
}

impl Intersects<&BoundingOrientedBox> for BoundingSphere {
    /// Test the BoundingSphere for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(constboundingorientedbox_)>
    fn Intersects(&self, box_: &BoundingOrientedBox) -> bool { 
        return box_.Intersects(self);
    }
}

impl Intersects<&BoundingFrustum> for BoundingSphere {
    /// Test the BoundingSphere for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(constboundingfrustum_)>
    fn Intersects(&self, fr: &BoundingFrustum) -> bool { 
        return fr.Intersects(self);
    }
}

impl Intersects<Triangle> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (V0, V1, V2): Triangle) -> bool { 
        // Load the sphere.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);

        // Compute the plane of the triangle (has to be normalized).
        let N: XMVECTOR = XMVector3Normalize(XMVector3Cross(XMVectorSubtract(V1, V0), XMVectorSubtract(V2, V0)));

        // Assert that the triangle is not degenerate.
        debug_assert!(!XMVector3Equal(N, XMVectorZero()));

        // Find the nearest feature on the triangle to the sphere.
        let Dist: XMVECTOR = XMVector3Dot(XMVectorSubtract(vCenter, V0), N);

        // If the center of the sphere is farther from the plane of the triangle than
        // the radius of the sphere, then there cannot be an intersection.
        let mut NoIntersection: XMVECTOR = XMVectorLess(Dist, XMVectorNegate(vRadius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Dist, vRadius));

        // Project the center of the sphere onto the plane of the triangle.
        let mut Point: XMVECTOR = XMVectorNegativeMultiplySubtract(N, Dist, vCenter);

        // Is it inside all the edges? If so we intersect because the distance
        // to the plane is less than the radius.
        let mut Intersection: XMVECTOR = internal::PointOnPlaneInsideTriangle(Point, V0, V1, V2);

        // Find the nearest point on each edge.
        let RadiusSq: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        // Edge 0,1
        Point = internal::PointOnLineSegmentNearestPoint(V0, V1, vCenter);

        // If the distance to the center of the sphere to the point is less than
        // the radius of the sphere then it must intersect.
        Intersection = XMVectorOrInt(Intersection, XMVectorLessOrEqual(XMVector3LengthSq(XMVectorSubtract(vCenter, Point)), RadiusSq));

        // Edge 1,2
        Point = internal::PointOnLineSegmentNearestPoint(V1, V2, vCenter);

        // If the distance to the center of the sphere to the point is less than
        // the radius of the sphere then it must intersect.
        Intersection = XMVectorOrInt(Intersection, XMVectorLessOrEqual(XMVector3LengthSq(XMVectorSubtract(vCenter, Point)), RadiusSq));

        // Edge 2,0
        Point = internal::PointOnLineSegmentNearestPoint(V2, V0, vCenter);

        // If the distance to the center of the sphere to the point is less than
        // the radius of the sphere then it must intersect.
        Intersection = XMVectorOrInt(Intersection, XMVectorLessOrEqual(XMVector3LengthSq(XMVectorSubtract(vCenter, Point)), RadiusSq));

        return XMVector4EqualInt(XMVectorAndCInt(Intersection, NoIntersection), XMVectorTrueInt());        
    }
}

impl Intersects<Plane, PlaneIntersectionType> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(fxmvector)>
    fn Intersects(&self, Plane: Plane) -> PlaneIntersectionType { 
        debug_assert!(internal::XMPlaneIsUnit(Plane));

        // Load the sphere.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);

        // Set w of the center to one so we can dot4 with a plane.
        // TODO: template
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        let mut Outside: XMVECTOR = unsafe { uninitialized() };
        let mut Inside: XMVECTOR = unsafe { uninitialized() };
        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane, &mut Outside, &mut Inside);

        // If the sphere is outside any plane it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return FRONT;
        }

        // If the sphere is inside all planes it is inside.
        if (XMVector4EqualInt(Inside, XMVectorTrueInt())) {
            return BACK;
        }

        // The sphere is not inside all planes or outside a plane it intersects.
        return INTERSECTING;
    }
}

pub type RayMut<'a> = (Point, Direction, &'a mut f32);

impl Intersects<RayMut<'_>> for BoundingSphere {
    /// Tests the BoundingSphere for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (Origin, Direction, Dist): RayMut<'_>) -> bool { 
        debug_assert!(internal::XMVector3IsUnit(Direction));

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);

        // l is the vector from the ray origin to the center of the sphere.
        let l: XMVECTOR = XMVectorSubtract(vCenter, Origin);

        // s is the projection of the l onto the ray direction.
        let s: XMVECTOR = XMVector3Dot(l, Direction);

        let l2: XMVECTOR = XMVector3Dot(l, l);

        let r2: XMVECTOR = XMVectorMultiply(vRadius, vRadius);

        // m2 is squared distance from the center of the sphere to the projection.
        let m2: XMVECTOR = XMVectorNegativeMultiplySubtract(s, s, l2);

        let mut NoIntersection: XMVECTOR;

        // If the ray origin is outside the sphere and the center of the sphere is
        // behind the ray origin there is no intersection.
        NoIntersection = XMVectorAndInt(XMVectorLess(s, XMVectorZero()), XMVectorGreater(l2, r2));

        // If the squared distance from the center of the sphere to the projection
        // is greater than the radius squared the ray will miss the sphere.
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(m2, r2));

        // The ray hits the sphere, compute the nearest intersection point.
        let q: XMVECTOR = XMVectorSqrt(XMVectorSubtract(r2, m2));
        let t1: XMVECTOR = XMVectorSubtract(s, q);
        let t2: XMVECTOR = XMVectorAdd(s, q);

        let OriginInside: XMVECTOR = XMVectorLessOrEqual(l2, r2);
        let t: XMVECTOR = XMVectorSelect(t1, t2, OriginInside);

        if (XMVector4NotEqualInt(NoIntersection, XMVectorTrueInt()))
        {
            // Store the x-component to *pDist.
            XMStoreFloat(Dist, t);
            return true;
        }

        *Dist = 0.0;
        return false;
    }
}

impl ContainedBy for BoundingSphere {
    /// Tests whether the BoundingSphere is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-containedby>
    fn ContainedBy(
        &self,
        Plane0: FXMVECTOR,
        Plane1: FXMVECTOR,
        Plane2: GXMVECTOR,
        Plane3: HXMVECTOR,
        Plane4: HXMVECTOR,
        Plane5: HXMVECTOR,
    ) -> ContainmentType
    {
        // Load the sphere.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&self.Radius);

        // Set w of the center to one so we can dot4 with a plane.
        // TODO: template
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        let mut Outside: XMVECTOR = unsafe { uninitialized() };
        let mut Inside: XMVECTOR = unsafe { uninitialized() };

        // Test against each plane.
        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane0, &mut Outside, &mut Inside);

        let mut AnyOutside: XMVECTOR = Outside;
        let mut AllInside: XMVECTOR = Inside;

        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane1, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane2, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane3, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane4, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectSpherePlane(vCenter, vRadius, Plane5, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        // If the sphere is outside any plane it is outside.
        if (XMVector4EqualInt(AnyOutside, XMVectorTrueInt())) {
            return DISJOINT;
        }

        // If the sphere is inside all planes it is inside.
        if (XMVector4EqualInt(AllInside, XMVectorTrueInt())) {
            return CONTAINS;
        }

        // The sphere is not inside all planes or outside a plane, it may intersect.
        return INTERSECTS;
    }
}

impl CreateMerged for BoundingSphere {
    /// Creates a BoundingSphere that contains the two specified BoundingSphere objects.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createmerged>
    fn CreateMerged(Out: &mut Self, S1: &Self, S2: &Self) {
        let Center1: XMVECTOR = XMLoadFloat3(&S1.Center);
        let r1: f32 = S1.Radius;

        let Center2: XMVECTOR = XMLoadFloat3(&S2.Center);
        let r2: f32 = S2.Radius;

        let V: XMVECTOR = XMVectorSubtract(Center2, Center1);

        let Dist: XMVECTOR = XMVector3Length(V);

        let d: f32 = XMVectorGetX(Dist);

        if (r1 + r2 >= d)
        {
            if (r1 - r2 >= d)
            {
                *Out = *S1;
                return;
            }
            else if (r2 - r1 >= d)
            {
                *Out = *S2;
                return;
            }
        }

        let N: XMVECTOR = XMVectorDivide(V, Dist);

        let t1: f32 = XMMin(-r1, d - r2);
        let t2: f32 = XMMax(r1, d + r2);
        let t_5: f32 = (t2 - t1) * 0.5;

        let NCenter: XMVECTOR = XMVectorAdd(Center1, XMVectorMultiply(N, XMVectorReplicate(t_5 + t1)));

        XMStoreFloat3(&mut Out.Center, NCenter);
        Out.Radius = t_5;
    }
}


impl CreateFromBoundingBox for BoundingSphere {
    /// Creates a BoundingSphere containing the specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createfromboundingbox>
    fn CreateFromBoundingBox(Out: &mut Self, box_: &BoundingBox) {
        Out.Center = box_.Center;
        let vExtents: XMVECTOR = XMLoadFloat3(&box_.Extents);
        Out.Radius = XMVectorGetX(XMVector3Length(vExtents));
    }
}

impl CreateFromBoundingOrientedBox for BoundingSphere {
    /// Creates a BoundingSphere containing the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createfromboundingbox(boundingsphere__constboundingorientedbox_)>
    fn CreateFromBoundingBox(Out: &mut Self, box_: &BoundingOrientedBox) {
        // Bounding box orientation is irrelevant because a sphere is rotationally invariant
        Out.Center = box_.Center;
        let vExtents: XMVECTOR = XMLoadFloat3(&box_.Extents);
        Out.Radius = XMVectorGetX(XMVector3Length(vExtents));
    }
}

impl CreateFromPoints for BoundingSphere {
    /// Creates a new BoundingSphere from a list of points.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingsphere-createfrompoints>
    fn CreateFromPoints<'a>(Out: &mut Self, pPoints: impl Iterator<Item=&'a XMFLOAT3> + Clone) {
        // assert(Count > 0);
        // assert(pPoints);

        // Find the points with minimum and maximum x, y, and z
        // XMVECTOR MinX, MaxX, MinY, MaxY, MinZ, MaxZ;

        // MinX = MaxX = MinY = MaxY = MinZ = MaxZ = XMLoadFloat3(pPoints);

        let mut MinX = g_XMZero.v();
        let mut MaxX = g_XMZero.v();
        let mut MinY = g_XMZero.v();
        let mut MaxY = g_XMZero.v();
        let mut MinZ = g_XMZero.v();
        let mut MaxZ = g_XMZero.v();

        // NOTE: We clone the iterator because it's reused down below.
        for (i, pPoint) in pPoints.clone().enumerate()
        {
            // XMVECTOR Point = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(reinterpret_cast<const uint8_t*>(pPoints) + i * Stride));
            let Point = XMLoadFloat3(pPoint);

            if i == 0 {
                MinX = Point;
                MaxX = Point;
                MinY = Point;
                MaxY = Point;
                MinZ = Point;
                MaxZ = Point;
            }

            let px: f32 = XMVectorGetX(Point);
            let py: f32 = XMVectorGetY(Point);
            let pz: f32 = XMVectorGetZ(Point);

            if (px < XMVectorGetX(MinX)) {
                MinX = Point;
            }

            if (px > XMVectorGetX(MaxX)) {
                MaxX = Point;
            }

            if (py < XMVectorGetY(MinY)) {
                MinY = Point;
            }

            if (py > XMVectorGetY(MaxY)) {
                MaxY = Point;
            }

            if (pz < XMVectorGetZ(MinZ)) {
                MinZ = Point;
            }

            if (pz > XMVectorGetZ(MaxZ)) {
                MaxZ = Point;
            }
        }

        // Use the min/max pair that are farthest apart to form the initial sphere.
        let DeltaX: XMVECTOR = XMVectorSubtract(MaxX, MinX);
        let DistX: XMVECTOR = XMVector3Length(DeltaX);

        let DeltaY: XMVECTOR = XMVectorSubtract(MaxY, MinY);
        let DistY: XMVECTOR = XMVector3Length(DeltaY);

        let DeltaZ: XMVECTOR = XMVectorSubtract(MaxZ, MinZ);
        let DistZ: XMVECTOR = XMVector3Length(DeltaZ);

        let mut vCenter: XMVECTOR;
        let mut vRadius: XMVECTOR;

        if (XMVector3Greater(DistX, DistY))
        {
            if (XMVector3Greater(DistX, DistZ))
            {
                // Use min/max x.
                vCenter = XMVectorLerp(MaxX, MinX, 0.5);
                vRadius = XMVectorScale(DistX, 0.5);
            }
            else
            {
                // Use min/max z.
                vCenter = XMVectorLerp(MaxZ, MinZ, 0.5);
                vRadius = XMVectorScale(DistZ, 0.5);
            }
        }
        else // Y >= X
        {
            if (XMVector3Greater(DistY, DistZ))
            {
                // Use min/max y.
                vCenter = XMVectorLerp(MaxY, MinY, 0.5);
                vRadius = XMVectorScale(DistY, 0.5);
            }
            else
            {
                // Use min/max z.
                vCenter = XMVectorLerp(MaxZ, MinZ, 0.5);
                vRadius = XMVectorScale(DistZ, 0.5);
            }
        }

        // Add any points not inside the sphere.
        for pPoint in pPoints
        {
            let Point: XMVECTOR = XMLoadFloat3(pPoint);

            let Delta: XMVECTOR = XMVectorSubtract(Point, vCenter);

            let Dist: XMVECTOR = XMVector3Length(Delta);

            if (XMVector3Greater(Dist, vRadius))
            {
                // Adjust sphere to include the new point.
                vRadius = XMVectorScale(XMVectorAdd(vRadius, Dist), 0.5);
                vCenter = XMVectorAdd(vCenter, XMVectorMultiply(XMVectorSubtract(XMVectorReplicate(1.0), XMVectorDivide(vRadius, Dist)), Delta));
            }
        }

        XMStoreFloat3(&mut Out.Center, vCenter);
        XMStoreFloat(&mut Out.Radius, vRadius);
    }
}

#[test]
fn test_BoundingSphere_CreateFromPoints() {
    let points = [
        XMFLOAT3 { x:  1.0, y: 0.0, z:  1.0 },
        XMFLOAT3 { x: -1.0, y: 0.0, z: -1.0 },
    ];

    let mut out: BoundingSphere = unsafe { uninitialized() };
    BoundingSphere::CreateFromPoints(&mut out, points.iter());

    assert_eq!(0.0, out.Center.x);
    assert_eq!(0.0, out.Center.y);
    assert_eq!(0.0, out.Center.z);
    assert_eq!(2.0f32.sqrt(), out.Radius);

    let points = [];

    let mut out: BoundingSphere = unsafe { uninitialized() };
    BoundingSphere::CreateFromPoints(&mut out, points.iter());

    assert_eq!(0.0, out.Center.x);
    assert_eq!(0.0, out.Center.y);
    assert_eq!(0.0, out.Center.z);
    assert_eq!(0.0, out.Radius); // NOTE: The DirectXMath source asserts points.len > 0
}

impl CreateFromFrustum for BoundingSphere {
    /// Creates a BoundingSphere containing the specified BoundingFrustum.
    fn CreateFromFrustum(Out: &mut Self, fr: &BoundingFrustum) {
        let mut Corners: [XMFLOAT3; BoundingFrustum::CORNER_COUNT] = unsafe { uninitialized() };
        fr.GetCorners(&mut Corners);
        BoundingSphere::CreateFromPoints(Out, Corners.iter())
    }
}


// BoundingBox ----------------------------------------------------------------

impl BoundingBox {
    pub fn GetCorners(&self, Corners: &mut [XMFLOAT3; BoundingBox::CORNER_COUNT]) {
        // Load the box
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        for i in 0..BoundingBox::CORNER_COUNT
        {
            let C: XMVECTOR = XMVectorMultiplyAdd(vExtents, g_BoxOffset[i].v(), vCenter);
            XMStoreFloat3(&mut Corners[i], C);
        }
    }
}

impl MatrixTransform for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-transform>
    fn Transform(&self, Out: &mut Self, M: FXMMATRIX) {
        // Load center and extents.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        // Compute and transform the corners and find new min/max bounds.
        let mut Corner: XMVECTOR = XMVectorMultiplyAdd(vExtents, g_BoxOffset[0].v(), vCenter);
        Corner = XMVector3Transform(Corner, M);

        let mut Min: XMVECTOR = Corner;
        let mut Max: XMVECTOR = Corner;

        for i in 1 .. BoundingBox::CORNER_COUNT
        {
            Corner = XMVectorMultiplyAdd(vExtents, g_BoxOffset[i].v(), vCenter);
            Corner = XMVector3Transform(Corner, M);

            Min = XMVectorMin(Min, Corner);
            Max = XMVectorMax(Max, Corner);
        }

        // Store center and extents.
        XMStoreFloat3(&mut Out.Center, XMVectorScale(XMVectorAdd(Min, Max), 0.5));
        XMStoreFloat3(&mut Out.Extents, XMVectorScale(XMVectorSubtract(Max, Min), 0.5));
    }
}

impl DecomposedTransform for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-transform(BoundingBox__float_fxmvector_fxmvector)>
    fn Transform(&self, Out: &mut Self, Scale: f32, Rotation: FXMVECTOR, Translation: FXMVECTOR) {
        debug_assert!(internal::XMQuaternionIsUnit(Rotation));

        // Load center and extents.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        let VectorScale: XMVECTOR = XMVectorReplicate(Scale);

        // Compute and transform the corners and find new min/max bounds.
        let mut Corner: XMVECTOR = XMVectorMultiplyAdd(vExtents, g_BoxOffset[0].v(), vCenter);
        Corner = XMVectorAdd(XMVector3Rotate(XMVectorMultiply(Corner, VectorScale), Rotation), Translation);

        let mut Min: XMVECTOR = Corner;
        let mut Max: XMVECTOR = Corner;

        for i in 1 .. BoundingBox::CORNER_COUNT
        {
            Corner = XMVectorMultiplyAdd(vExtents, g_BoxOffset[i].v(), vCenter);
            Corner = XMVectorAdd(XMVector3Rotate(XMVectorMultiply(Corner, VectorScale), Rotation), Translation);

            Min = XMVectorMin(Min, Corner);
            Max = XMVectorMax(Max, Corner);
        }

        // Store center and extents.
        XMStoreFloat3(&mut Out.Center, XMVectorScale(XMVectorAdd(Min, Max), 0.5));
        XMStoreFloat3(&mut Out.Extents, XMVectorScale(XMVectorSubtract(Max, Min), 0.5));
    }
}

impl Contains<Point> for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains>
    fn Contains(&self, Point: Point) -> ContainmentType {
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        return if XMVector3InBounds(XMVectorSubtract(Point, vCenter), vExtents) { CONTAINS } else { DISJOINT };
    }
}

impl Contains<Triangle> for BoundingBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (V0, V1, V2): Triangle) -> ContainmentType {
        if (!self.Intersects((V0, V1, V2))) {
            return DISJOINT;
        }

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        let mut d: XMVECTOR = XMVectorAbs(XMVectorSubtract(V0, vCenter));
        let mut Inside: XMVECTOR = XMVectorLessOrEqual(d, vExtents);

        d = XMVectorAbs(XMVectorSubtract(V1, vCenter));
        Inside = XMVectorAndInt(Inside, XMVectorLessOrEqual(d, vExtents));

        d = XMVectorAbs(XMVectorSubtract(V2, vCenter));
        Inside = XMVectorAndInt(Inside, XMVectorLessOrEqual(d, vExtents));

        return if (XMVector3EqualInt(Inside, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingSphere> for BoundingBox {
    /// Tests whether the BoundingBox contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingsphere_)>
    fn Contains(&self, sh: &BoundingSphere) -> ContainmentType {
        let SphereCenter: XMVECTOR = XMLoadFloat3(&sh.Center);
        let SphereRadius: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);

        let BoxCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let BoxExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        let BoxMin: XMVECTOR = XMVectorSubtract(BoxCenter, BoxExtents);
        let BoxMax: XMVECTOR = XMVectorAdd(BoxCenter, BoxExtents);

        // Find the distance to the nearest point on the box.
        // for each i in (x, y, z)
        // if (SphereCenter(i) < BoxMin(i)) d2 += (SphereCenter(i) - BoxMin(i)) ^ 2
        // else if (SphereCenter(i) > BoxMax(i)) d2 += (SphereCenter(i) - BoxMax(i)) ^ 2

        let mut d: XMVECTOR = XMVectorZero();

        // Compute d for each dimension.
        let LessThanMin: XMVECTOR = XMVectorLess(SphereCenter, BoxMin);
        let GreaterThanMax: XMVECTOR = XMVectorGreater(SphereCenter, BoxMax);

        let MinDelta: XMVECTOR = XMVectorSubtract(SphereCenter, BoxMin);
        let MaxDelta: XMVECTOR = XMVectorSubtract(SphereCenter, BoxMax);

        // Choose value for each dimension based on the comparison.
        d = XMVectorSelect(d, MinDelta, LessThanMin);
        d = XMVectorSelect(d, MaxDelta, GreaterThanMax);

        // Use a dot-product to square them and sum them together.
        let d2: XMVECTOR = XMVector3Dot(d, d);

        if (XMVector3Greater(d2, XMVectorMultiply(SphereRadius, SphereRadius))) {
            return DISJOINT;
        }

        let mut InsideAll: XMVECTOR = XMVectorLessOrEqual(XMVectorAdd(BoxMin, SphereRadius), SphereCenter);
        InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(SphereCenter, XMVectorSubtract(BoxMax, SphereRadius)));
        InsideAll = XMVectorAndInt(InsideAll, XMVectorGreater(XMVectorSubtract(BoxMax, BoxMin), SphereRadius));

        return if (XMVector3EqualInt(InsideAll, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingBox> for BoundingBox {
    /// Tests whether the BoundingBox contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingbox_)>
    fn Contains(&self, box_: &BoundingBox) -> ContainmentType {
        let CenterA: XMVECTOR = XMLoadFloat3(&self.Center);
        let ExtentsA: XMVECTOR = XMLoadFloat3(&self.Extents);

        let CenterB: XMVECTOR = XMLoadFloat3(&box_.Center);
        let ExtentsB: XMVECTOR = XMLoadFloat3(&box_.Extents);

        let MinA: XMVECTOR = XMVectorSubtract(CenterA, ExtentsA);
        let MaxA: XMVECTOR = XMVectorAdd(CenterA, ExtentsA);

        let MinB: XMVECTOR = XMVectorSubtract(CenterB, ExtentsB);
        let MaxB: XMVECTOR = XMVectorAdd(CenterB, ExtentsB);

        // for each i in (x, y, z) if a_min(i) > b_max(i) or b_min(i) > a_max(i) then return false
        let Disjoint: XMVECTOR = XMVectorOrInt(XMVectorGreater(MinA, MaxB), XMVectorGreater(MinB, MaxA));

        if (internal::XMVector3AnyTrue(Disjoint)) {
            return DISJOINT;
        }

        // for each i in (x, y, z) if a_min(i) <= b_min(i) and b_max(i) <= a_max(i) then A contains B
        let Inside: XMVECTOR = XMVectorAndInt(XMVectorLessOrEqual(MinA, MinB), XMVectorLessOrEqual(MaxB, MaxA));

        return if internal::XMVector3AllTrue(Inside) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingOrientedBox> for BoundingBox {
    /// Tests whether the BoundingBox contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingorientedbox_)>
    fn Contains(&self, box_: &BoundingOrientedBox) -> ContainmentType {
        if (!box_.Intersects(self)) {
            return DISJOINT;
        }

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        // Subtract off the AABB center to remove a subtract below
        let oCenter: XMVECTOR = XMVectorSubtract(XMLoadFloat3(&box_.Center), vCenter);

        let oExtents: XMVECTOR = XMLoadFloat3(&box_.Extents);
        let oOrientation: XMVECTOR = XMLoadFloat4(&box_.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(oOrientation));

        let mut Inside: XMVECTOR = XMVectorTrueInt();

        //for (size_t i = 0; i < BoundingOrientedBox::CORNER_COUNT; ++i)
        for i in 0 .. BoundingOrientedBox::CORNER_COUNT
        {
            let C: XMVECTOR = XMVectorAdd(XMVector3Rotate(XMVectorMultiply(oExtents, g_BoxOffset[i].v()), oOrientation), oCenter);
            let d: XMVECTOR = XMVectorAbs(C);
            Inside = XMVectorAndInt(Inside, XMVectorLessOrEqual(d, vExtents));
        }

        return if (XMVector3EqualInt(Inside, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingFrustum> for BoundingBox {
    /// Tests whether the BoundingBox contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-contains(constboundingfrustum_)>
    fn Contains(&self, fr: &BoundingFrustum) -> ContainmentType {
        if (!fr.Intersects(self)) {
            return DISJOINT;
        }

        let mut Corners: [XMFLOAT3; BoundingFrustum::CORNER_COUNT] = unsafe { uninitialized() };
        fr.GetCorners(&mut Corners);

        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        let mut Inside: XMVECTOR = XMVectorTrueInt();

        for i in 0 ..BoundingFrustum::CORNER_COUNT
        {
            let Point: XMVECTOR = XMLoadFloat3(&Corners[i]);
            let d: XMVECTOR = XMVectorAbs(XMVectorSubtract(Point, vCenter));
            Inside = XMVectorAndInt(Inside, XMVectorLessOrEqual(d, vExtents));
        }

        return if (XMVector3EqualInt(Inside, XMVectorTrueInt())) { CONTAINS } else { INTERSECTS };
    }
}

impl Intersects<&BoundingSphere> for BoundingBox {
    /// Tests the BoundingBox for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects>
    fn Intersects(&self, sh: &BoundingSphere) -> bool {
        let SphereCenter: XMVECTOR = XMLoadFloat3(&sh.Center);
        let SphereRadius: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);
    
        let BoxCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let BoxExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
    
        let BoxMin: XMVECTOR = XMVectorSubtract(BoxCenter, BoxExtents);
        let BoxMax: XMVECTOR = XMVectorAdd(BoxCenter, BoxExtents);
    
        // Find the distance to the nearest point on the box.
        // for each i in (x, y, z)
        // if (SphereCenter(i) < BoxMin(i)) d2 += (SphereCenter(i) - BoxMin(i)) ^ 2
        // else if (SphereCenter(i) > BoxMax(i)) d2 += (SphereCenter(i) - BoxMax(i)) ^ 2
    
        let mut d: XMVECTOR = XMVectorZero();
    
        // Compute d for each dimension.
        let LessThanMin: XMVECTOR = XMVectorLess(SphereCenter, BoxMin);
        let GreaterThanMax: XMVECTOR = XMVectorGreater(SphereCenter, BoxMax);
    
        let MinDelta: XMVECTOR = XMVectorSubtract(SphereCenter, BoxMin);
        let MaxDelta: XMVECTOR = XMVectorSubtract(SphereCenter, BoxMax);
    
        // Choose value for each dimension based on the comparison.
        d = XMVectorSelect(d, MinDelta, LessThanMin);
        d = XMVectorSelect(d, MaxDelta, GreaterThanMax);
    
        // Use a dot-product to square them and sum them together.
        let d2: XMVECTOR = XMVector3Dot(d, d);
    
        return XMVector3LessOrEqual(d2, XMVectorMultiply(SphereRadius, SphereRadius));
    }
}

impl Intersects<&BoundingBox> for BoundingBox {
    /// Tests the BoundingBox for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(constboundingbox_)>
    fn Intersects(&self, box_: &BoundingBox) -> bool {
        let CenterA: XMVECTOR = XMLoadFloat3(&self.Center);
        let ExtentsA: XMVECTOR = XMLoadFloat3(&self.Extents);

        let CenterB: XMVECTOR = XMLoadFloat3(&box_.Center);
        let ExtentsB: XMVECTOR = XMLoadFloat3(&box_.Extents);

        let MinA: XMVECTOR = XMVectorSubtract(CenterA, ExtentsA);
        let MaxA: XMVECTOR = XMVectorAdd(CenterA, ExtentsA);

        let MinB: XMVECTOR = XMVectorSubtract(CenterB, ExtentsB);
        let MaxB: XMVECTOR = XMVectorAdd(CenterB, ExtentsB);

        // for each i in (x, y, z) if a_min(i) > b_max(i) or b_min(i) > a_max(i) then return false
        let Disjoint: XMVECTOR = XMVectorOrInt(XMVectorGreater(MinA, MaxB), XMVectorGreater(MinB, MaxA));

        return !internal::XMVector3AnyTrue(Disjoint);
    }
}

impl Intersects<&BoundingOrientedBox> for BoundingBox {
    /// Test the BoundingBox for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(constboundingorientedbox_)>
    fn Intersects(&self, box_: &BoundingOrientedBox) -> bool {
        return box_.Intersects(self);
    }
}

impl Intersects<&BoundingFrustum> for BoundingBox {
    /// Test the BoundingBox for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(constboundingfrustum_)>
    fn Intersects(&self, fr: &BoundingFrustum) -> bool {
        return fr.Intersects(self);
    }
}

impl Intersects<Triangle> for BoundingBox {
    /// Tests the BoundingSphere for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (V0, V1, V2): Triangle) -> bool {
        let Zero: XMVECTOR = XMVectorZero();

        // Load the box.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        let BoxMin: XMVECTOR = XMVectorSubtract(vCenter, vExtents);
        let BoxMax: XMVECTOR = XMVectorAdd(vCenter, vExtents);

        // Test the axes of the box (in effect test the AAB against the minimal AAB
        // around the triangle).
        let TriMin: XMVECTOR = XMVectorMin(XMVectorMin(V0, V1), V2);
        let TriMax: XMVECTOR = XMVectorMax(XMVectorMax(V0, V1), V2);

        // for each i in (x, y, z) if a_min(i) > b_max(i) or b_min(i) > a_max(i) then disjoint
        let Disjoint: XMVECTOR = XMVectorOrInt(XMVectorGreater(TriMin, BoxMax), XMVectorGreater(BoxMin, TriMax));
        if (internal::XMVector3AnyTrue(Disjoint)) {
            return false;
        }

        // Test the plane of the triangle.
        let Normal: XMVECTOR = XMVector3Cross(XMVectorSubtract(V1, V0), XMVectorSubtract(V2, V0));
        let Dist: XMVECTOR = XMVector3Dot(Normal, V0);

        // Assert that the triangle is not degenerate.
        debug_assert!(!XMVector3Equal(Normal, Zero));

        // for each i in (x, y, z) if n(i) >= 0 then v_min(i)=b_min(i), v_max(i)=b_max(i)
        // else v_min(i)=b_max(i), v_max(i)=b_min(i)
        let NormalSelect: XMVECTOR = XMVectorGreater(Normal, Zero);
        let V_Min: XMVECTOR = XMVectorSelect(BoxMax, BoxMin, NormalSelect);
        let V_Max: XMVECTOR = XMVectorSelect(BoxMin, BoxMax, NormalSelect);

        // if n dot v_min + d > 0 || n dot v_max + d < 0 then disjoint
        let MinDist: XMVECTOR = XMVector3Dot(V_Min, Normal);
        let MaxDist: XMVECTOR = XMVector3Dot(V_Max, Normal);

        let mut NoIntersection: XMVECTOR = XMVectorGreater(MinDist, Dist);
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(MaxDist, Dist));

        // Move the box center to zero to simplify the following tests.
        let TV0: XMVECTOR = XMVectorSubtract(V0, vCenter);
        let TV1: XMVECTOR = XMVectorSubtract(V1, vCenter);
        let TV2: XMVECTOR = XMVectorSubtract(V2, vCenter);

        // Test the edge/edge axes (3*3).
        let mut e0: XMVECTOR = XMVectorSubtract(TV1, TV0);
        let mut e1: XMVECTOR = XMVectorSubtract(TV2, TV1);
        let mut e2: XMVECTOR = XMVectorSubtract(TV0, TV2);

        // Make w zero.
        // TODO: template
        e0 = XMVectorInsert(e0, Zero, 0, 0, 0, 0, 1);
        e1 = XMVectorInsert(e1, Zero, 0, 0, 0, 0, 1);
        e2 = XMVectorInsert(e2, Zero, 0, 0, 0, 0, 1);

        let mut Axis: XMVECTOR;
        let mut p0: XMVECTOR;
        let mut p1: XMVECTOR;
        let mut p2: XMVECTOR;
        let mut Min: XMVECTOR;
        let mut Max: XMVECTOR;
        let mut Radius: XMVECTOR;

        // Axis == (1,0,0) let e0: x = (0, -e0.z, e0.y)
        Axis = <(XM_PERMUTE_0W, XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_0X)>::XMVectorPermute(e0, XMVectorNegate(e0));
        p0 = XMVector3Dot(TV0, Axis);
        // p1 = XMVector3Dot( V1, Axis ); // p1 = p0;
        p2 = XMVector3Dot(TV2, Axis);
        Min = XMVectorMin(p0, p2);
        Max = XMVectorMax(p0, p2);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (1,0,0) let e1: x = (0, -e1.z, e1.y)
        Axis = <(XM_PERMUTE_0W, XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_0X)>::XMVectorPermute(e1, XMVectorNegate(e1));
        p0 = XMVector3Dot(TV0, Axis);
        p1 = XMVector3Dot(TV1, Axis);
        // p2 = XMVector3Dot( V2, Axis ); // p2 = p1;
        Min = XMVectorMin(p0, p1);
        Max = XMVectorMax(p0, p1);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (1,0,0) let e2: x = (0, -e2.z, e2.y)
        Axis = <(XM_PERMUTE_0W, XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_0X)>::XMVectorPermute(e2, XMVectorNegate(e2));
        p0 = XMVector3Dot(TV0, Axis);
        p1 = XMVector3Dot(TV1, Axis);
        // p2 = XMVector3Dot( V2, Axis ); // p2 = p0;
        Min = XMVectorMin(p0, p1);
        Max = XMVectorMax(p0, p1);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (0,1,0) let e0: x = (e0.z, 0, -e0.x)
        Axis = <(XM_PERMUTE_0Z, XM_PERMUTE_0W, XM_PERMUTE_1X, XM_PERMUTE_0Y)>::XMVectorPermute(e0, XMVectorNegate(e0));
        p0 = XMVector3Dot(TV0, Axis);
        // p1 = XMVector3Dot( V1, Axis ); // p1 = p0;
        p2 = XMVector3Dot(TV2, Axis);
        Min = XMVectorMin(p0, p2);
        Max = XMVectorMax(p0, p2);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (0,1,0) let e1: x = (e1.z, 0, -e1.x)
        Axis = <(XM_PERMUTE_0Z, XM_PERMUTE_0W, XM_PERMUTE_1X, XM_PERMUTE_0Y)>::XMVectorPermute(e1, XMVectorNegate(e1));
        p0 = XMVector3Dot(TV0, Axis);
        p1 = XMVector3Dot(TV1, Axis);
        // p2 = XMVector3Dot( V2, Axis ); // p2 = p1;
        Min = XMVectorMin(p0, p1);
        Max = XMVectorMax(p0, p1);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (0,0,1) let e2: x = (e2.z, 0, -e2.x)
        Axis = <(XM_PERMUTE_0Z, XM_PERMUTE_0W, XM_PERMUTE_1X, XM_PERMUTE_0Y)>::XMVectorPermute(e2, XMVectorNegate(e2));
        p0 = XMVector3Dot(TV0, Axis);
        p1 = XMVector3Dot(TV1, Axis);
        // p2 = XMVector3Dot( V2, Axis ); // p2 = p0;
        Min = XMVectorMin(p0, p1);
        Max = XMVectorMax(p0, p1);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (0,0,1) let e0: x = (-e0.y, e0.x, 0)
        Axis = <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_0Z)>::XMVectorPermute(e0, XMVectorNegate(e0));
        p0 = XMVector3Dot(TV0, Axis);
        // p1 = XMVector3Dot( V1, Axis ); // p1 = p0;
        p2 = XMVector3Dot(TV2, Axis);
        Min = XMVectorMin(p0, p2);
        Max = XMVectorMax(p0, p2);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (0,0,1) let e1: x = (-e1.y, e1.x, 0)
        Axis = <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_0Z)>::XMVectorPermute(e1, XMVectorNegate(e1));
        p0 = XMVector3Dot(TV0, Axis);
        p1 = XMVector3Dot(TV1, Axis);
        // p2 = XMVector3Dot( V2, Axis ); // p2 = p1;
        Min = XMVectorMin(p0, p1);
        Max = XMVectorMax(p0, p1);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        // Axis == (0,0,1) let e2: x = (-e2.y, e2.x, 0)
        Axis = <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_0Z)>::XMVectorPermute(e2, XMVectorNegate(e2));
        p0 = XMVector3Dot(TV0, Axis);
        p1 = XMVector3Dot(TV1, Axis);
        // p2 = XMVector3Dot( V2, Axis ); // p2 = p0;
        Min = XMVectorMin(p0, p1);
        Max = XMVectorMax(p0, p1);
        Radius = XMVector3Dot(vExtents, XMVectorAbs(Axis));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(Min, Radius));
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(Max, XMVectorNegate(Radius)));

        return XMVector4NotEqualInt(NoIntersection, XMVectorTrueInt());
    }
}

impl Intersects<Plane, PlaneIntersectionType> for BoundingBox {
    /// Tests the BoundingBox for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(fxmvector)>
    fn Intersects(&self, Plane: Plane) -> PlaneIntersectionType {
        debug_assert!(internal::XMPlaneIsUnit(Plane));

        // Load the box.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        // Set w of the center to one so we can dot4 with a plane.
        // TODO: template
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        let mut Outside: XMVECTOR = unsafe { uninitialized() };
        let mut Inside: XMVECTOR = unsafe { uninitialized() };
        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane, &mut Outside, &mut Inside);

        // If the box is outside any plane it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return FRONT;
        }

        // If the box is inside all planes it is inside.
        if (XMVector4EqualInt(Inside, XMVectorTrueInt())) {
            return BACK;
        }

        // The box is not inside all planes or outside a plane it intersects.
        return INTERSECTING;
    }
}

impl Intersects<RayMut<'_>> for BoundingBox {
    /// Tests the BoundingBox for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (Origin, Direction, Dist): RayMut<'_>) -> bool {
        debug_assert!(internal::XMVector3IsUnit(Direction));

        // Load the box.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        // Adjust ray origin to be relative to center of the box.
        let TOrigin: XMVECTOR = XMVectorSubtract(vCenter, Origin);

        // Compute the dot product againt each axis of the box.
        // Since the axii are (1,0,0), (0,1,0), (0,0,1) no computation is necessary.
        let AxisDotOrigin: XMVECTOR = TOrigin;
        let AxisDotDirection: XMVECTOR = Direction;

        // if (fabs(AxisDotDirection) <= Epsilon) the ray is nearly parallel to the slab.
        let IsParallel: XMVECTOR = XMVectorLessOrEqual(XMVectorAbs(AxisDotDirection), g_RayEpsilon.v());

        // Test against all three axii simultaneously.
        let InverseAxisDotDirection: XMVECTOR = XMVectorReciprocal(AxisDotDirection);
        let t1: XMVECTOR = XMVectorMultiply(XMVectorSubtract(AxisDotOrigin, vExtents), InverseAxisDotDirection);
        let t2: XMVECTOR = XMVectorMultiply(XMVectorAdd(AxisDotOrigin, vExtents), InverseAxisDotDirection);

        // Compute the max of min(t1,t2) and the min of max(t1,t2) ensuring we don't
        // use the results from any directions parallel to the slab.
        let mut t_min: XMVECTOR = XMVectorSelect(XMVectorMin(t1, t2), g_FltMin.v(), IsParallel);
        let mut t_max: XMVECTOR = XMVectorSelect(XMVectorMax(t1, t2), g_FltMax.v(), IsParallel);

        // t_min.x = maximum( t_min.x, t_min.y, t_min.z );
        // t_max.x = minimum( t_max.x, t_max.y, t_max.z );
        t_min = XMVectorMax(t_min, XMVectorSplatY(t_min));  // x = max(x,y)
        t_min = XMVectorMax(t_min, XMVectorSplatZ(t_min));  // x = max(max(x,y),z)
        t_max = XMVectorMin(t_max, XMVectorSplatY(t_max));  // x = min(x,y)
        t_max = XMVectorMin(t_max, XMVectorSplatZ(t_max));  // x = min(min(x,y),z)

        // if ( t_min > t_max ) return false;
        let mut NoIntersection: XMVECTOR = XMVectorGreater(XMVectorSplatX(t_min), XMVectorSplatX(t_max));

        // if ( t_max < 0.0f ) return false;
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(XMVectorSplatX(t_max), XMVectorZero()));

        // if (IsParallel && (-Extents > AxisDotOrigin || Extents < AxisDotOrigin)) return false;
        let ParallelOverlap: XMVECTOR = XMVectorInBounds(AxisDotOrigin, vExtents);
        NoIntersection = XMVectorOrInt(NoIntersection, XMVectorAndCInt(IsParallel, ParallelOverlap));

        if (!internal::XMVector3AnyTrue(NoIntersection))
        {
            // Store the x-component to *pDist
            XMStoreFloat(Dist, t_min);
            return true;
        }

        *Dist = 0.0;
        return false;
    }
}

impl ContainedBy for BoundingBox {
    /// Tests whether the BoundingBox is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-containedby>
    fn ContainedBy(
        &self,
        Plane0: FXMVECTOR,
        Plane1: FXMVECTOR,
        Plane2: GXMVECTOR,
        Plane3: HXMVECTOR,
        Plane4: HXMVECTOR,
        Plane5: HXMVECTOR,
    ) -> ContainmentType
    {
        // Load the box.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);

        // Set w of the center to one so we can dot4 with a plane.
        // TODO: template
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        let mut Outside: XMVECTOR = unsafe { uninitialized() };
        let mut Inside: XMVECTOR = unsafe { uninitialized() };

        // Test against each plane.
        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane0, &mut Outside, &mut Inside);

        let mut AnyOutside: XMVECTOR = Outside;
        let mut AllInside: XMVECTOR = Inside;

        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane1, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane2, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane3, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane4, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectAxisAlignedBoxPlane(vCenter, vExtents, Plane5, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        // If the box is outside any plane it is outside.
        if (XMVector4EqualInt(AnyOutside, XMVectorTrueInt())) {
            return DISJOINT;
        }

        // If the box is inside all planes it is inside.
        if (XMVector4EqualInt(AllInside, XMVectorTrueInt())) {
            return CONTAINS;
        }

        // The box is not inside all planes or outside a plane, it may intersect.
        return INTERSECTS;
    }
}

impl CreateMerged for BoundingBox {
    /// Creates a BoundingBox that contains the two specified BoundingBox objects.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-createmerged>
    fn CreateMerged(Out: &mut Self, b1: &Self, b2: &Self) {
        let b1Center: XMVECTOR = XMLoadFloat3(&b1.Center);
        let b1Extents: XMVECTOR = XMLoadFloat3(&b1.Extents);

        let b2Center: XMVECTOR = XMLoadFloat3(&b2.Center);
        let b2Extents: XMVECTOR = XMLoadFloat3(&b2.Extents);

        let mut Min: XMVECTOR = XMVectorSubtract(b1Center, b1Extents);
        Min = XMVectorMin(Min, XMVectorSubtract(b2Center, b2Extents));

        let mut Max: XMVECTOR = XMVectorAdd(b1Center, b1Extents);
        Max = XMVectorMax(Max, XMVectorAdd(b2Center, b2Extents));

        debug_assert!(XMVector3LessOrEqual(Min, Max));

        XMStoreFloat3(&mut Out.Center, XMVectorScale(XMVectorAdd(Min, Max), 0.5));
        XMStoreFloat3(&mut Out.Extents, XMVectorScale(XMVectorSubtract(Max, Min), 0.5));
    }
}

impl CreateFromSphere for BoundingBox {
    /// Creates a BoundingBox large enough to contain the a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-boundingbox-createfromsphere>
    fn CreateFromSphere(Out: &mut Self, sh: &BoundingSphere) {
        let spCenter: XMVECTOR = XMLoadFloat3(&sh.Center);
        let shRadius: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);

        let Min: XMVECTOR = XMVectorSubtract(spCenter, shRadius);
        let Max: XMVECTOR = XMVectorAdd(spCenter, shRadius);

        debug_assert!(XMVector3LessOrEqual(Min, Max));

        XMStoreFloat3(&mut Out.Center, XMVectorScale(XMVectorAdd(Min, Max), 0.5));
        XMStoreFloat3(&mut Out.Extents, XMVectorScale(XMVectorSubtract(Max, Min), 0.5));
    }
}

impl CreateFromPoints for BoundingBox {
    /// Creates a new BoundingBox from a list of points.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingBox-createfrompoints>
    fn CreateFromPoints<'a>(Out: &mut Self, pPoints: impl Iterator<Item=&'a XMFLOAT3>) {
        // assert(Count > 0);
        // assert(pPoints);

        // Find the minimum and maximum x, y, and z
        // NOTE: We default to Zero since we don't have the Count > 0 assertion
        let mut vMin: XMVECTOR = g_XMZero.v();
        let mut vMax: XMVECTOR = g_XMZero.v();

        for (i, pPoint) in pPoints.enumerate()
        {
            let Point: XMVECTOR = XMLoadFloat3(pPoint);
            if i == 0 {
                vMin = Point;
                vMax = Point;
            } else {
                vMin = XMVectorMin(vMin, Point);
                vMax = XMVectorMax(vMax, Point);
            }
        }

        // Store center and extents.
        XMStoreFloat3(&mut Out.Center, XMVectorScale(XMVectorAdd(vMin, vMax), 0.5));
        XMStoreFloat3(&mut Out.Extents, XMVectorScale(XMVectorSubtract(vMax, vMin), 0.5));
    }
}

// BoundingOrientedBox --------------------------------------------------------

impl MatrixTransform for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-transform>
    fn Transform(&self, Out: &mut Self, M: FXMMATRIX) {
        // Load the box.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let mut vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let mut vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        unsafe {
            // Composite the box rotation and the transform rotation.
            let mut nM: XMMATRIX = uninitialized();
            nM.r[0] = XMVector3Normalize(M.r[0]);
            nM.r[1] = XMVector3Normalize(M.r[1]);
            nM.r[2] = XMVector3Normalize(M.r[2]);
            nM.r[3] = g_XMIdentityR3.v;
            let Rotation: XMVECTOR = XMQuaternionRotationMatrix(nM);
            vOrientation = XMQuaternionMultiply(vOrientation, Rotation);

            // Transform the center.
            vCenter = XMVector3Transform(vCenter, M);

            // Scale the box extents.
            let dX: XMVECTOR = XMVector3Length(M.r[0]);
            let dY: XMVECTOR = XMVector3Length(M.r[1]);
            let dZ: XMVECTOR = XMVector3Length(M.r[2]);

            let mut VectorScale: XMVECTOR = XMVectorSelect(dY, dX, g_XMSelect1000.v);
            VectorScale = XMVectorSelect(dZ, VectorScale, g_XMSelect1100.v);
            vExtents = XMVectorMultiply(vExtents, VectorScale);
        }

        // Store the box.
        XMStoreFloat3(&mut Out.Center, vCenter);
        XMStoreFloat3(&mut Out.Extents, vExtents);
        XMStoreFloat4(&mut Out.Orientation, vOrientation);
    }
}

impl DecomposedTransform for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-transform(BoundingOrientedBox__float_fxmvector_fxmvector)>
    fn Transform(&self, Out: &mut Self, Scale: f32, Rotation: FXMVECTOR, Translation: FXMVECTOR) {
        debug_assert!(internal::XMQuaternionIsUnit(Rotation));

        // Load the box.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let mut vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let mut vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        // Composite the box rotation and the transform rotation.
        vOrientation = XMQuaternionMultiply(vOrientation, Rotation);

        // Transform the center.
        let VectorScale: XMVECTOR = XMVectorReplicate(Scale);
        vCenter = XMVectorAdd(XMVector3Rotate(XMVectorMultiply(vCenter, VectorScale), Rotation), Translation);

        // Scale the box extents.
        vExtents = XMVectorMultiply(vExtents, VectorScale);

        // Store the box.
        XMStoreFloat3(&mut Out.Center, vCenter);
        XMStoreFloat3(&mut Out.Extents, vExtents);
        XMStoreFloat4(&mut Out.Orientation, vOrientation);
    }
}

impl Contains<Point> for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains>
    fn Contains(&self, Point: Point) -> ContainmentType {
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Transform the point to be local to the box.
        let TPoint: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(Point, vCenter), vOrientation);

        return if XMVector3InBounds(TPoint, vExtents) { CONTAINS } else { DISJOINT };
    }
}

impl Contains<Triangle> for BoundingOrientedBox {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (V0, V1, V2): Triangle) -> ContainmentType {
        // Load the box center & orientation.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Transform the triangle vertices into the space of the box.
        let TV0: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V0, vCenter), vOrientation);
        let TV1: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V1, vCenter), vOrientation);
        let TV2: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V2, vCenter), vOrientation);

        let mut box_: BoundingBox = unsafe { uninitialized() };
        box_.Center = XMFLOAT3::set(0.0, 0.0, 0.0);
        box_.Extents = self.Extents;

        // Use the triangle vs axis aligned box intersection routine.
        return box_.Contains((TV0, TV1, TV2));
    }
}

impl Contains<&BoundingSphere> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingsphere_)>
    fn Contains(&self, sh: &BoundingSphere) -> ContainmentType {
        let mut SphereCenter: XMVECTOR = XMLoadFloat3(&sh.Center);
        let SphereRadius: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);

        let BoxCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let BoxExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let BoxOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(BoxOrientation));

        // Transform the center of the sphere to be local to the box.
        // BoxMin = -BoxExtents
        // BoxMax = +BoxExtents
        SphereCenter = XMVector3InverseRotate(XMVectorSubtract(SphereCenter, BoxCenter), BoxOrientation);

        // Find the distance to the nearest point on the box.
        // for each i in (x, y, z)
        // if (SphereCenter(i) < BoxMin(i)) d2 += (SphereCenter(i) - BoxMin(i)) ^ 2
        // else if (SphereCenter(i) > BoxMax(i)) d2 += (SphereCenter(i) - BoxMax(i)) ^ 2

        let mut d: XMVECTOR = XMVectorZero();

        // Compute d for each dimension.
        let LessThanMin: XMVECTOR = XMVectorLess(SphereCenter, XMVectorNegate(BoxExtents));
        let GreaterThanMax: XMVECTOR = XMVectorGreater(SphereCenter, BoxExtents);

        let MinDelta: XMVECTOR = XMVectorAdd(SphereCenter, BoxExtents);
        let MaxDelta: XMVECTOR = XMVectorSubtract(SphereCenter, BoxExtents);

        // Choose value for each dimension based on the comparison.
        d = XMVectorSelect(d, MinDelta, LessThanMin);
        d = XMVectorSelect(d, MaxDelta, GreaterThanMax);

        // Use a dot-product to square them and sum them together.
        let d2: XMVECTOR = XMVector3Dot(d, d);
        let SphereRadiusSq: XMVECTOR = XMVectorMultiply(SphereRadius, SphereRadius);

        if (XMVector4Greater(d2, SphereRadiusSq)) {
            return DISJOINT;
        }

        // See if we are completely inside the box
        let SMin: XMVECTOR = XMVectorSubtract(SphereCenter, SphereRadius);
        let SMax: XMVECTOR = XMVectorAdd(SphereCenter, SphereRadius);

        return if (XMVector3InBounds(SMin, BoxExtents) && XMVector3InBounds(SMax, BoxExtents)) { CONTAINS } else { INTERSECTS };
    }
}

impl Contains<&BoundingBox> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingbox_)>
    fn Contains(&self, box_: &BoundingBox) -> ContainmentType {
        // Make the axis aligned box oriented and do an OBB vs OBB test.
        let obox = BoundingOrientedBox {
            Center: box_.Center,
            Extents: box_.Extents,
            Orientation: XMFLOAT4::set(0.0, 0.0, 0.0, 1.0),
        };
        return self.Contains(&obox);
    }
}

impl Contains<&BoundingOrientedBox> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingorientedbox_)>
    fn Contains(&self, box_: &BoundingOrientedBox) -> ContainmentType {
        if (!self.Intersects(box_)) {
            return DISJOINT;
        }

        // Load the boxes
        let aCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let aExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let aOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(aOrientation));

        let bCenter: XMVECTOR = XMLoadFloat3(&box_.Center);
        let bExtents: XMVECTOR = XMLoadFloat3(&box_.Extents);
        let bOrientation: XMVECTOR = XMLoadFloat4(&box_.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(bOrientation));

        let offset: XMVECTOR = XMVectorSubtract(bCenter, aCenter);

        for i in 0..BoundingOrientedBox::CORNER_COUNT
        {
            // Cb = rotate( bExtents * corneroffset[i], bOrientation ) + bcenter
            // Ca = invrotate( Cb - aCenter, aOrientation )

            let mut C: XMVECTOR = XMVectorAdd(XMVector3Rotate(XMVectorMultiply(bExtents, g_BoxOffset[i].v()), bOrientation), offset);
            C = XMVector3InverseRotate(C, aOrientation);

            if (!XMVector3InBounds(C, aExtents)) {
                return INTERSECTS;
            }
        }

        return CONTAINS;
    }
}

impl Contains<&BoundingFrustum> for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-contains(constboundingfrustum_)>
    fn Contains(&self, fr: &BoundingFrustum) -> ContainmentType {
        if (!fr.Intersects(self)) {
            return DISJOINT;
        }

        let mut Corners: [XMFLOAT3; BoundingFrustum::CORNER_COUNT] = unsafe { uninitialized() };
        fr.GetCorners(&mut Corners);

        // Load the box
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        for i in 0 .. BoundingFrustum::CORNER_COUNT
        {
            let C: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(XMLoadFloat3(&Corners[i]), vCenter), vOrientation);

            if (!XMVector3InBounds(C, vExtents)) {
                return INTERSECTS;
            }
        }

        return CONTAINS;
    }
}

impl Intersects<&BoundingSphere> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects>
    fn Intersects(&self, sh: &BoundingSphere) -> bool {
        let mut SphereCenter: XMVECTOR = XMLoadFloat3(&sh.Center);
        let SphereRadius: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);

        let BoxCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let BoxExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let BoxOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(BoxOrientation));

        // Transform the center of the sphere to be local to the box.
        // BoxMin = -BoxExtents
        // BoxMax = +BoxExtents
        SphereCenter = XMVector3InverseRotate(XMVectorSubtract(SphereCenter, BoxCenter), BoxOrientation);

        // Find the distance to the nearest point on the box.
        // for each i in (x, y, z)
        // if (SphereCenter(i) < BoxMin(i)) d2 += (SphereCenter(i) - BoxMin(i)) ^ 2
        // else if (SphereCenter(i) > BoxMax(i)) d2 += (SphereCenter(i) - BoxMax(i)) ^ 2

        let mut d: XMVECTOR = XMVectorZero();

        // Compute d for each dimension.
        let LessThanMin: XMVECTOR = XMVectorLess(SphereCenter, XMVectorNegate(BoxExtents));
        let GreaterThanMax: XMVECTOR = XMVectorGreater(SphereCenter, BoxExtents);

        let MinDelta: XMVECTOR = XMVectorAdd(SphereCenter, BoxExtents);
        let MaxDelta: XMVECTOR = XMVectorSubtract(SphereCenter, BoxExtents);

        // Choose value for each dimension based on the comparison.
        d = XMVectorSelect(d, MinDelta, LessThanMin);
        d = XMVectorSelect(d, MaxDelta, GreaterThanMax);

        // Use a dot-product to square them and sum them together.
        let d2: XMVECTOR = XMVector3Dot(d, d);

        return if XMVector4LessOrEqual(d2, XMVectorMultiply(SphereRadius, SphereRadius)) { true } else { false };
    }
}

impl Intersects<&BoundingBox> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(constboundingbox_)>
    fn Intersects(&self, box_: &BoundingBox) -> bool {
        // Make the axis aligned box oriented and do an OBB vs OBB test.
        let obox = BoundingOrientedBox {
            Center: box_.Center,
            Extents: box_.Extents,
            Orientation: XMFLOAT4::set(0.0, 0.0, 0.0, 1.0),
        };
        return self.Intersects(&obox);
    }
}

impl Intersects<&BoundingOrientedBox> for BoundingOrientedBox {
    /// Test the BoundingOrientedBox for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(constboundingorientedbox_)>
    fn Intersects(&self, box_: &BoundingOrientedBox) -> bool {
        // Build the 3x3 rotation matrix that defines the orientation of B relative to A.
        let A_quat: XMVECTOR = XMLoadFloat4(&self.Orientation);
        let B_quat: XMVECTOR = XMLoadFloat4(&box_.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(A_quat));
        debug_assert!(internal::XMQuaternionIsUnit(B_quat));

        let Q: XMVECTOR = XMQuaternionMultiply(A_quat, XMQuaternionConjugate(B_quat));
        let R: XMMATRIX = XMMatrixRotationQuaternion(Q);

        // Compute the translation of B relative to A.
        let A_cent: XMVECTOR = XMLoadFloat3(&self.Center);
        let B_cent: XMVECTOR = XMLoadFloat3(&box_.Center);
        let t: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(B_cent, A_cent), A_quat);

        //
        // h(A) = extents of A.
        // h(B) = extents of B.
        //
        // a(u) = axes let A: of = (1,0,0), (0,1,0), (0,0,1)
        // b(u) = axes of B relative let A: to = (r00,r10,r20), (r01,r11,r21), (r02,r12,r22)
        //
        // For each possible separating axis l:
        //   d(A) = sum (let i: for = u,v,w) h(A)(i) * abs( a(i) dot l )
        //   d(B) = sum (let i: for = u,v,w) h(B)(i) * abs( b(i) dot l )
        //   if abs( t dot l ) > d(A) + d(B) then disjoint
        //

        // Load extents of A and B.
        let h_A: XMVECTOR = XMLoadFloat3(&self.Extents);
        let h_B: XMVECTOR = XMLoadFloat3(&box_.Extents);

        // Rows. Note R[0,1,2]X.w = 0.
        let R0X: XMVECTOR = unsafe { R.r[0] };
        let R1X: XMVECTOR = unsafe { R.r[1] };
        let R2X: XMVECTOR = unsafe { R.r[2] };

        let R = XMMatrixTranspose(R);

        // Columns. Note RX[0,1,2].w = 0.
        let RX0: XMVECTOR = unsafe { R.r[0] };
        let RX1: XMVECTOR = unsafe { R.r[1] };
        let RX2: XMVECTOR = unsafe { R.r[2] };

        // Absolute value of rows.
        let AR0X: XMVECTOR = XMVectorAbs(R0X);
        let AR1X: XMVECTOR = XMVectorAbs(R1X);
        let AR2X: XMVECTOR = XMVectorAbs(R2X);

        // Absolute value of columns.
        let ARX0: XMVECTOR = XMVectorAbs(RX0);
        let ARX1: XMVECTOR = XMVectorAbs(RX1);
        let ARX2: XMVECTOR = XMVectorAbs(RX2);

        // Test each of the 15 possible seperating axii.
        //XMVECTOR d, d_A, d_B;
        let mut d: XMVECTOR;
        let mut d_A: XMVECTOR;
        let mut d_B: XMVECTOR;

        // l = a(u) = (1, 0, 0)
        // t let l: dot = t.x
        // d(A) = h(A).x
        // d(B) = h(B) dot abs(r00, r01, r02)
        d = XMVectorSplatX(t);
        d_A = XMVectorSplatX(h_A);
        d_B = XMVector3Dot(h_B, AR0X);
        let mut NoIntersection: XMVECTOR = XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B));

        // l = a(v) = (0, 1, 0)
        // t let l: dot = t.y
        // d(A) = h(A).y
        // d(B) = h(B) dot abs(r10, r11, r12)
        d = XMVectorSplatY(t);
        d_A = XMVectorSplatY(h_A);
        d_B = XMVector3Dot(h_B, AR1X);
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(w) = (0, 0, 1)
        // t let l: dot = t.z
        // d(A) = h(A).z
        // d(B) = h(B) dot abs(r20, r21, r22)
        d = XMVectorSplatZ(t);
        d_A = XMVectorSplatZ(h_A);
        d_B = XMVector3Dot(h_B, AR2X);
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = b(u) = (r00, r10, r20)
        // d(A) = h(A) dot abs(r00, r10, r20)
        // d(B) = h(B).x
        d = XMVector3Dot(t, RX0);
        d_A = XMVector3Dot(h_A, ARX0);
        d_B = XMVectorSplatX(h_B);
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = b(v) = (r01, r11, r21)
        // d(A) = h(A) dot abs(r01, r11, r21)
        // d(B) = h(B).y
        d = XMVector3Dot(t, RX1);
        d_A = XMVector3Dot(h_A, ARX1);
        d_B = XMVectorSplatY(h_B);
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = b(w) = (r02, r12, r22)
        // d(A) = h(A) dot abs(r02, r12, r22)
        // d(B) = h(B).z
        d = XMVector3Dot(t, RX2);
        d_A = XMVector3Dot(h_A, ARX2);
        d_B = XMVectorSplatZ(h_B);
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(u) x b(u) = (0, -r20, r10)
        // d(A) = h(A) dot abs(0, r20, r10)
        // d(B) = h(B) dot abs(0, r02, r01)
        d = XMVector3Dot(t, <(XM_PERMUTE_0W, XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_0X)>::XMVectorPermute(RX0, XMVectorNegate(RX0)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(ARX0));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(AR0X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(u) x b(v) = (0, -r21, r11)
        // d(A) = h(A) dot abs(0, r21, r11)
        // d(B) = h(B) dot abs(r02, 0, r00)
        d = XMVector3Dot(t, <(XM_PERMUTE_0W, XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_0X)>::XMVectorPermute(RX1, XMVectorNegate(RX1)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(ARX1));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(AR0X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(u) x b(w) = (0, -r22, r12)
        // d(A) = h(A) dot abs(0, r22, r12)
        // d(B) = h(B) dot abs(r01, r00, 0)
        d = XMVector3Dot(t, <(XM_PERMUTE_0W, XM_PERMUTE_1Z, XM_PERMUTE_0Y, XM_PERMUTE_0X)>::XMVectorPermute(RX2, XMVectorNegate(RX2)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(ARX2));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(AR0X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(v) x b(u) = (r20, 0, -r00)
        // d(A) = h(A) dot abs(r20, 0, r00)
        // d(B) = h(B) dot abs(0, r12, r11)
        d = XMVector3Dot(t, <(XM_PERMUTE_0Z, XM_PERMUTE_0W, XM_PERMUTE_1X, XM_PERMUTE_0Y)>::XMVectorPermute(RX0, XMVectorNegate(RX0)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(ARX0));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(AR1X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(v) x b(v) = (r21, 0, -r01)
        // d(A) = h(A) dot abs(r21, 0, r01)
        // d(B) = h(B) dot abs(r12, 0, r10)
        d = XMVector3Dot(t, <(XM_PERMUTE_0Z, XM_PERMUTE_0W, XM_PERMUTE_1X, XM_PERMUTE_0Y)>::XMVectorPermute(RX1, XMVectorNegate(RX1)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(ARX1));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(AR1X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(v) x b(w) = (r22, 0, -r02)
        // d(A) = h(A) dot abs(r22, 0, r02)
        // d(B) = h(B) dot abs(r11, r10, 0)
        d = XMVector3Dot(t, <(XM_PERMUTE_0Z, XM_PERMUTE_0W, XM_PERMUTE_1X, XM_PERMUTE_0Y)>::XMVectorPermute(RX2, XMVectorNegate(RX2)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(ARX2));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(AR1X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(w) x b(u) = (-r10, r00, 0)
        // d(A) = h(A) dot abs(r10, r00, 0)
        // d(B) = h(B) dot abs(0, r22, r21)
        d = XMVector3Dot(t, <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_0Z)>::XMVectorPermute(RX0, XMVectorNegate(RX0)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(ARX0));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_W, XM_SWIZZLE_Z, XM_SWIZZLE_Y, XM_SWIZZLE_X)>::XMVectorSwizzle(AR2X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(w) x b(v) = (-r11, r01, 0)
        // d(A) = h(A) dot abs(r11, r01, 0)
        // d(B) = h(B) dot abs(r22, 0, r20)
        d = XMVector3Dot(t, <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_0Z)>::XMVectorPermute(RX1, XMVectorNegate(RX1)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(ARX1));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_Z, XM_SWIZZLE_W, XM_SWIZZLE_X, XM_SWIZZLE_Y)>::XMVectorSwizzle(AR2X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // l = a(w) x b(w) = (-r12, r02, 0)
        // d(A) = h(A) dot abs(r12, r02, 0)
        // d(B) = h(B) dot abs(r21, r20, 0)
        d = XMVector3Dot(t, <(XM_PERMUTE_1Y, XM_PERMUTE_0X, XM_PERMUTE_0W, XM_PERMUTE_0Z)>::XMVectorPermute(RX2, XMVectorNegate(RX2)));
        d_A = XMVector3Dot(h_A, <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(ARX2));
        d_B = XMVector3Dot(h_B, <(XM_SWIZZLE_Y, XM_SWIZZLE_X, XM_SWIZZLE_W, XM_SWIZZLE_Z)>::XMVectorSwizzle(AR2X));
        NoIntersection = XMVectorOrInt(NoIntersection,
            XMVectorGreater(XMVectorAbs(d), XMVectorAdd(d_A, d_B)));

        // No seperating axis found, boxes must intersect.
        return if XMVector4NotEqualInt(NoIntersection, XMVectorTrueInt()) { true } else { false };
    }
}

impl Intersects<&BoundingFrustum> for BoundingOrientedBox {
    /// Test the BoundingOrientedBox for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(constboundingfrustum_)>
    fn Intersects(&self, fr: &BoundingFrustum) -> bool {
        return fr.Intersects(self);
    }
}

impl Intersects<Triangle> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (V0, V1, V2): Triangle) -> bool {
        // Load the box center & orientation.
        let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Transform the triangle vertices into the space of the box.
        let TV0: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V0, vCenter), vOrientation);
        let TV1: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V1, vCenter), vOrientation);
        let TV2: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V2, vCenter), vOrientation);

        let mut box_: BoundingBox = unsafe { uninitialized() };
        box_.Center = XMFLOAT3::set(0.0, 0.0, 0.0);
        box_.Extents = self.Extents;

        // Use the triangle vs axis aligned box intersection routine.
        return box_.Intersects((TV0, TV1, TV2));
    }
}

impl Intersects<Plane, PlaneIntersectionType> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(fxmvector)>
    fn Intersects(&self, Plane: Plane) -> PlaneIntersectionType {
        debug_assert!(internal::XMPlaneIsUnit(Plane));

        // Load the box.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let BoxOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);
    
        debug_assert!(internal::XMQuaternionIsUnit(BoxOrientation));
    
        // Set w of the center to one so we can dot4 with a plane.
        // TODO: template
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);
    
        // Build the 3x3 rotation matrix that defines the box axes.
        let R: XMMATRIX = XMMatrixRotationQuaternion(BoxOrientation);
    
        unsafe {
            let mut Outside: XMVECTOR = uninitialized();
            let mut Inside: XMVECTOR = uninitialized();
            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane, &mut Outside, &mut Inside);
        
            // If the box is outside any plane it is outside.
            if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
                return FRONT;
            }
        
            // If the box is inside all planes it is inside.
            if (XMVector4EqualInt(Inside, XMVectorTrueInt())) {
                return BACK;
            }
        }
    
        // The box is not inside all planes or outside a plane it intersects.
        return INTERSECTING;        
    }
}

impl Intersects<RayMut<'_>> for BoundingOrientedBox {
    /// Tests the BoundingOrientedBox for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (Origin, Direction, Dist): RayMut<'_>) -> bool {
        unsafe {
            debug_assert!(internal::XMVector3IsUnit(Direction));

            const SelectY: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_1, XM_SELECT_0, XM_SELECT_0 ] };
            const SelectZ: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_0, XM_SELECT_1, XM_SELECT_0 ] };

            // Load the box.
            let vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
            let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
            let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

            debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

            // Get the boxes normalized side directions.
            let R: XMMATRIX = XMMatrixRotationQuaternion(vOrientation);

            // Adjust ray origin to be relative to center of the box.
            let TOrigin: XMVECTOR = XMVectorSubtract(vCenter, Origin);

            // Compute the dot product againt each axis of the box.
            let mut AxisDotOrigin: XMVECTOR = XMVector3Dot(R.r[0], TOrigin);
            AxisDotOrigin = XMVectorSelect(AxisDotOrigin, XMVector3Dot(R.r[1], TOrigin), SelectY.v);
            AxisDotOrigin = XMVectorSelect(AxisDotOrigin, XMVector3Dot(R.r[2], TOrigin), SelectZ.v);

            let mut AxisDotDirection: XMVECTOR = XMVector3Dot(R.r[0], Direction);
            AxisDotDirection = XMVectorSelect(AxisDotDirection, XMVector3Dot(R.r[1], Direction), SelectY.v);
            AxisDotDirection = XMVectorSelect(AxisDotDirection, XMVector3Dot(R.r[2], Direction), SelectZ.v);

            // if (fabs(AxisDotDirection) <= Epsilon) the ray is nearly parallel to the slab.
            let IsParallel: XMVECTOR = XMVectorLessOrEqual(XMVectorAbs(AxisDotDirection), g_RayEpsilon.v);

            // Test against all three axes simultaneously.
            let InverseAxisDotDirection: XMVECTOR = XMVectorReciprocal(AxisDotDirection);
            let t1: XMVECTOR = XMVectorMultiply(XMVectorSubtract(AxisDotOrigin, vExtents), InverseAxisDotDirection);
            let t2: XMVECTOR = XMVectorMultiply(XMVectorAdd(AxisDotOrigin, vExtents), InverseAxisDotDirection);

            // Compute the max of min(t1,t2) and the min of max(t1,t2) ensuring we don't
            // use the results from any directions parallel to the slab.
            let mut t_min: XMVECTOR = XMVectorSelect(XMVectorMin(t1, t2), g_FltMin.v, IsParallel);
            let mut t_max: XMVECTOR = XMVectorSelect(XMVectorMax(t1, t2), g_FltMax.v, IsParallel);

            // t_min.x = maximum( t_min.x, t_min.y, t_min.z );
            // t_max.x = minimum( t_max.x, t_max.y, t_max.z );
            t_min = XMVectorMax(t_min, XMVectorSplatY(t_min));  // x = max(x,y)
            t_min = XMVectorMax(t_min, XMVectorSplatZ(t_min));  // x = max(max(x,y),z)
            t_max = XMVectorMin(t_max, XMVectorSplatY(t_max));  // x = min(x,y)
            t_max = XMVectorMin(t_max, XMVectorSplatZ(t_max));  // x = min(min(x,y),z)

            // if ( t_min > t_max ) return false;
            let mut NoIntersection: XMVECTOR = XMVectorGreater(XMVectorSplatX(t_min), XMVectorSplatX(t_max));

            // if ( t_max < 0.0f ) return false;
            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(XMVectorSplatX(t_max), XMVectorZero()));

            // if (IsParallel && (-Extents > AxisDotOrigin || Extents < AxisDotOrigin)) return false;
            let ParallelOverlap: XMVECTOR = XMVectorInBounds(AxisDotOrigin, vExtents);
            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorAndCInt(IsParallel, ParallelOverlap));

            if (!internal::XMVector3AnyTrue(NoIntersection))
            {
                // Store the x-component to *pDist
                XMStoreFloat(Dist, t_min);
                return true;
            }

            *Dist = 0.0;
            return false;
        }
    }
}

impl ContainedBy for BoundingOrientedBox {
    /// Tests whether the BoundingOrientedBox is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-containedby>
    fn ContainedBy(
        &self,
        Plane0: FXMVECTOR,
        Plane1: FXMVECTOR,
        Plane2: GXMVECTOR,
        Plane3: HXMVECTOR,
        Plane4: HXMVECTOR,
        Plane5: HXMVECTOR,
    ) -> ContainmentType
    {
        // Load the box.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&self.Center);
        let vExtents: XMVECTOR = XMLoadFloat3(&self.Extents);
        let BoxOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(BoxOrientation));

        // Set w of the center to one so we can dot4 with a plane.
        // TODO: template
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        // Build the 3x3 rotation matrix that defines the box axes.
        let R: XMMATRIX = XMMatrixRotationQuaternion(BoxOrientation);

        unsafe { 
            let mut Outside: XMVECTOR = uninitialized();
            let mut Inside: XMVECTOR = uninitialized();

            // Test against each plane.
            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane0, &mut Outside, &mut Inside);

            let mut AnyOutside: XMVECTOR = Outside;
            let mut AllInside: XMVECTOR = Inside;

            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane1, &mut Outside, &mut Inside);
            AnyOutside = XMVectorOrInt(AnyOutside, Outside);
            AllInside = XMVectorAndInt(AllInside, Inside);

            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane2, &mut Outside, &mut Inside);
            AnyOutside = XMVectorOrInt(AnyOutside, Outside);
            AllInside = XMVectorAndInt(AllInside, Inside);

            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane3, &mut Outside, &mut Inside);
            AnyOutside = XMVectorOrInt(AnyOutside, Outside);
            AllInside = XMVectorAndInt(AllInside, Inside);

            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane4, &mut Outside, &mut Inside);
            AnyOutside = XMVectorOrInt(AnyOutside, Outside);
            AllInside = XMVectorAndInt(AllInside, Inside);

            internal::FastIntersectOrientedBoxPlane(vCenter, vExtents, R.r[0], R.r[1], R.r[2], Plane5, &mut Outside, &mut Inside);
            AnyOutside = XMVectorOrInt(AnyOutside, Outside);
            AllInside = XMVectorAndInt(AllInside, Inside);

            // If the box is outside any plane it is outside.
            if (XMVector4EqualInt(AnyOutside, XMVectorTrueInt())) {
                return DISJOINT;
            }

            // If the box is inside all planes it is inside.
            if (XMVector4EqualInt(AllInside, XMVectorTrueInt())) {
                return CONTAINS;
            }
        }

        // The box is not inside all planes or outside a plane, it may intersect.
        return INTERSECTS;
    }
}

// Create oriented bounding box from axis-aligned bounding box
impl CreateFromBoundingBox for BoundingOrientedBox {
    /// Creates a BoundingBox large enough to contain the a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-CreateFromBoundingBox>
    fn CreateFromBoundingBox(Out: &mut Self, box_: &BoundingBox) {
        Out.Center = box_.Center;
        Out.Extents = box_.Extents;
        Out.Orientation = XMFLOAT4::set(0.0, 0.0, 0.0, 1.0);
    }
}

//-----------------------------------------------------------------------------
// Find the approximate minimum oriented bounding box containing a set of
// points.  Exact computation of minimum oriented bounding box is possible but
// is slower and requires a more complex algorithm.
// The algorithm works by computing the inertia tensor of the points and then
// using the eigenvectors of the intertia tensor as the axes of the box.
// Computing the intertia tensor of the convex hull of the points will usually
// result in better bounding box but the computation is more complex.
// Exact computation of the minimum oriented bounding box is possible but the
// best know algorithm is O(N^3) and is significanly more complex to implement.
//-----------------------------------------------------------------------------
impl CreateFromPoints for BoundingOrientedBox {
    /// Creates a new BoundingOrientedBox from a list of points.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingOrientedBox-createfrompoints>
    fn CreateFromPoints<'a>(Out: &mut Self, pPoints: impl Iterator<Item=&'a XMFLOAT3> + Clone) {
        // TODO: Determine the best way to handle an empty set of points

        // assert(Count > 0);
        // assert(pPoints != nullptr);

        let mut CenterOfMass: XMVECTOR = XMVectorZero();

        // Compute the center of mass and inertia tensor of the points.
        //for (let i: size_t = 0; i < Count; ++i)
        for point in pPoints.clone()
        {
            //let Point: XMVECTOR = XMLoadFloat3(reinterpret_cast<const XMFLOAT3*>(reinterpret_cast<const uint8_t*>(pPoints) + i * Stride));
            let Point: XMVECTOR = XMLoadFloat3(point);

            CenterOfMass = XMVectorAdd(CenterOfMass, Point);
        }

        // TODO: ExactSizeIterator
        let Count = pPoints.clone().count();

        CenterOfMass = XMVectorMultiply(CenterOfMass, XMVectorReciprocal(XMVectorReplicate(Count as f32)));

        // Compute the inertia tensor of the points around the center of mass.
        // Using the center of mass is not strictly necessary, but will hopefully
        // improve the stability of finding the eigenvectors.
        let mut XX_YY_ZZ: XMVECTOR = XMVectorZero();
        let mut XY_XZ_YZ: XMVECTOR = XMVectorZero();

        //for (let i: size_t = 0; i < Count; ++i)
        for point in pPoints.clone()
        {
            let Point: XMVECTOR = XMVectorSubtract(XMLoadFloat3(point), CenterOfMass);

            XX_YY_ZZ = XMVectorAdd(XX_YY_ZZ, XMVectorMultiply(Point, Point));

            let XXY: XMVECTOR = <(XM_SWIZZLE_X, XM_SWIZZLE_X, XM_SWIZZLE_Y, XM_SWIZZLE_W)>::XMVectorSwizzle(Point);
            let YZZ: XMVECTOR = <(XM_SWIZZLE_Y, XM_SWIZZLE_Z, XM_SWIZZLE_Z, XM_SWIZZLE_W)>::XMVectorSwizzle(Point);

            XY_XZ_YZ = XMVectorAdd(XY_XZ_YZ, XMVectorMultiply(XXY, YZZ));
        }

        let mut v1: XMVECTOR = unsafe { uninitialized() };
        let mut v2: XMVECTOR = unsafe { uninitialized() };
        let mut v3: XMVECTOR = unsafe { uninitialized() };

        // Compute the eigenvectors of the inertia tensor.
        internal::CalculateEigenVectorsFromCovarianceMatrix(XMVectorGetX(XX_YY_ZZ), XMVectorGetY(XX_YY_ZZ),
            XMVectorGetZ(XX_YY_ZZ),
            XMVectorGetX(XY_XZ_YZ), XMVectorGetY(XY_XZ_YZ),
            XMVectorGetZ(XY_XZ_YZ),
            &mut v1, &mut v2, &mut v3);

        // Put them in a matrix.
        let mut R: XMMATRIX = unsafe { uninitialized() };

        unsafe {
            R.r[0] = XMVectorSetW(v1, 0.0);
            R.r[1] = XMVectorSetW(v2, 0.0);
            R.r[2] = XMVectorSetW(v3, 0.0);
            R.r[3] = g_XMIdentityR3.v;
        }

        // Multiply by -1 to convert the matrix into a right handed coordinate
        // system (Det ~= 1) in case the eigenvectors form a left handed
        // coordinate system (Det ~= -1) because XMQuaternionRotationMatrix only
        // works on right handed matrices.
        let Det: XMVECTOR = XMMatrixDeterminant(R);

        if (XMVector4Less(Det, XMVectorZero()))
        {
            unsafe {
                R.r[0] = XMVectorMultiply(R.r[0], g_XMNegativeOne.v);
                R.r[1] = XMVectorMultiply(R.r[1], g_XMNegativeOne.v);
                R.r[2] = XMVectorMultiply(R.r[2], g_XMNegativeOne.v);
            }
        }

        // Get the rotation quaternion from the matrix.
        let mut vOrientation: XMVECTOR = XMQuaternionRotationMatrix(R);

        // Make sure it is normal (in case the vectors are slightly non-orthogonal).
        vOrientation = XMQuaternionNormalize(vOrientation);

        // Rebuild the rotation matrix from the quaternion.
        R = XMMatrixRotationQuaternion(vOrientation);

        // Build the rotation into the rotated space.
        let InverseR: XMMATRIX = XMMatrixTranspose(R);

        // Find the minimum OBB using the eigenvectors as the axes.
        let mut vMin: XMVECTOR = XMVectorZero();
        let mut vMax: XMVECTOR = XMVectorZero();

        //for (let i: size_t = 1; i < Count; ++i)
        for (i, point) in pPoints.clone().enumerate()
        {
            let Point: XMVECTOR = XMLoadFloat3(point);

            if i == 0 {
                vMin = XMVector3TransformNormal(Point, InverseR);
                vMax = vMin;
            } else {
                vMin = XMVectorMin(vMin, Point);
                vMax = XMVectorMax(vMax, Point);
            }
        }

        // Rotate the center into world space.
        let mut vCenter: XMVECTOR = XMVectorScale(XMVectorAdd(vMin, vMax), 0.5);
        vCenter = XMVector3TransformNormal(vCenter, R);

        // Store center, extents, and orientation.
        XMStoreFloat3(&mut Out.Center, vCenter);
        XMStoreFloat3(&mut Out.Extents, XMVectorScale(XMVectorSubtract(vMax, vMin), 0.5));
        XMStoreFloat4(&mut Out.Orientation, vOrientation);
    }
}

// BoundingFrustum ----------------------------------------------------------------

impl MatrixTransform for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-transform>
    fn Transform(&self, Out: &mut Self, M: FXMMATRIX) {
        // Load the frustum.
        let mut vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let mut vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        unsafe {
            // Composite the frustum rotation and the transform rotation
            let mut nM: XMMATRIX = uninitialized();
            nM.r[0] = XMVector3Normalize(M.r[0]);
            nM.r[1] = XMVector3Normalize(M.r[1]);
            nM.r[2] = XMVector3Normalize(M.r[2]);
            nM.r[3] = g_XMIdentityR3.v;
            let Rotation: XMVECTOR = XMQuaternionRotationMatrix(nM);
            vOrientation = XMQuaternionMultiply(vOrientation, Rotation);

            // Transform the center.
            vOrigin = XMVector3Transform(vOrigin, M);

            // Store the frustum.
            XMStoreFloat3(&mut Out.Origin, vOrigin);
            XMStoreFloat4(&mut Out.Orientation, vOrientation);

            // Scale the near and far distances (the slopes remain the same).
            let dX: XMVECTOR = XMVector3Dot(M.r[0], M.r[0]);
            let dY: XMVECTOR = XMVector3Dot(M.r[1], M.r[1]);
            let dZ: XMVECTOR = XMVector3Dot(M.r[2], M.r[2]);

            let d: XMVECTOR = XMVectorMax(dX, XMVectorMax(dY, dZ));
            let Scale: f32 = sqrtf(XMVectorGetX(d));

            Out.Near = self.Near * Scale;
            Out.Far = self.Far * Scale;
        }

        // Copy the slopes.
        Out.RightSlope = self.RightSlope;
        Out.LeftSlope = self.LeftSlope;
        Out.TopSlope = self.TopSlope;
        Out.BottomSlope = self.BottomSlope;
    }
}

impl DecomposedTransform for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-transform(BoundingFrustum__float_fxmvector_fxmvector)>
    fn Transform(&self, Out: &mut Self, Scale: f32, Rotation: FXMVECTOR, Translation: FXMVECTOR) {
        debug_assert!(internal::XMQuaternionIsUnit(Rotation));

        // Load the frustum.
        let mut vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let mut vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        // Composite the frustum rotation and the transform rotation.
        vOrientation = XMQuaternionMultiply(vOrientation, Rotation);

        // Transform the origin.
        vOrigin = XMVectorAdd(XMVector3Rotate(XMVectorScale(vOrigin, Scale), Rotation), Translation);

        // Store the frustum.
        XMStoreFloat3(&mut Out.Origin, vOrigin);
        XMStoreFloat4(&mut Out.Orientation, vOrientation);

        // Scale the near and far distances (the slopes remain the same).
        Out.Near = self.Near * Scale;
        Out.Far = self.Far * Scale;

        // Copy the slopes.
        Out.RightSlope = self.RightSlope;
        Out.LeftSlope = self.LeftSlope;
        Out.TopSlope = self.TopSlope;
        Out.BottomSlope = self.BottomSlope;
    }
}

impl Contains<Point> for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains>
    fn Contains(&self, Point: Point) -> ContainmentType {
        // Build frustum planes.
        let mut Planes: [XMVECTOR; 6] = unsafe { uninitialized() };
        Planes[0] = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        Planes[1] = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        Planes[2] = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        Planes[3] = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        Planes[4] = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        Planes[5] = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);

        // Load origin and orientation.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        // Transform point into local space of frustum.
        let mut TPoint: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(Point, vOrigin), vOrientation);

        // Set w to one.
        // TODO: template
        TPoint = XMVectorInsert(TPoint, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        let Zero: XMVECTOR = XMVectorZero();
        let mut Outside: XMVECTOR = Zero;

        // Test point against each plane of the frustum.
        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            let Dot: XMVECTOR = XMVector4Dot(TPoint, Planes[i]);
            Outside = XMVectorOrInt(Outside, XMVectorGreater(Dot, Zero));
        }

        return if XMVector4NotEqualInt(Outside, XMVectorTrueInt()) { CONTAINS } else { DISJOINT };
    }
}

impl Contains<Triangle> for BoundingFrustum {
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(fxmvector_fxmvector_fxmvector)>
    fn Contains(&self, (V0, V1, V2): Triangle) -> ContainmentType {
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Create 6 planes (do it inline to encourage use of registers)
        let mut NearPlane: XMVECTOR = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        NearPlane = internal::XMPlaneTransform(NearPlane, vOrientation, vOrigin);
        NearPlane = XMPlaneNormalize(NearPlane);

        let mut FarPlane: XMVECTOR = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        FarPlane = internal::XMPlaneTransform(FarPlane, vOrientation, vOrigin);
        FarPlane = XMPlaneNormalize(FarPlane);

        let mut RightPlane: XMVECTOR = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        RightPlane = internal::XMPlaneTransform(RightPlane, vOrientation, vOrigin);
        RightPlane = XMPlaneNormalize(RightPlane);

        let mut LeftPlane: XMVECTOR = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        LeftPlane = internal::XMPlaneTransform(LeftPlane, vOrientation, vOrigin);
        LeftPlane = XMPlaneNormalize(LeftPlane);

        let mut TopPlane: XMVECTOR = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        TopPlane = internal::XMPlaneTransform(TopPlane, vOrientation, vOrigin);
        TopPlane = XMPlaneNormalize(TopPlane);

        let mut BottomPlane: XMVECTOR = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
        BottomPlane = internal::XMPlaneTransform(BottomPlane, vOrientation, vOrigin);
        BottomPlane = XMPlaneNormalize(BottomPlane);

        return triangle_tests::ContainedBy(V0, V1, V2, NearPlane, FarPlane, RightPlane, &LeftPlane, &TopPlane, &BottomPlane);
    }
}

impl Contains<&BoundingSphere> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains a specified BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingsphere_)>
    fn Contains(&self, sh: &BoundingSphere) -> ContainmentType {
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Create 6 planes (do it inline to encourage use of registers)
        let mut NearPlane: XMVECTOR = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        NearPlane = internal::XMPlaneTransform(NearPlane, vOrientation, vOrigin);
        NearPlane = XMPlaneNormalize(NearPlane);

        let mut FarPlane: XMVECTOR = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        FarPlane = internal::XMPlaneTransform(FarPlane, vOrientation, vOrigin);
        FarPlane = XMPlaneNormalize(FarPlane);

        let mut RightPlane: XMVECTOR = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        RightPlane = internal::XMPlaneTransform(RightPlane, vOrientation, vOrigin);
        RightPlane = XMPlaneNormalize(RightPlane);

        let mut LeftPlane: XMVECTOR = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        LeftPlane = internal::XMPlaneTransform(LeftPlane, vOrientation, vOrigin);
        LeftPlane = XMPlaneNormalize(LeftPlane);

        let mut TopPlane: XMVECTOR = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        TopPlane = internal::XMPlaneTransform(TopPlane, vOrientation, vOrigin);
        TopPlane = XMPlaneNormalize(TopPlane);

        let mut BottomPlane: XMVECTOR = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
        BottomPlane = internal::XMPlaneTransform(BottomPlane, vOrientation, vOrigin);
        BottomPlane = XMPlaneNormalize(BottomPlane);

        return sh.ContainedBy(NearPlane, FarPlane, RightPlane, LeftPlane, TopPlane, BottomPlane);
    }
}

impl Contains<&BoundingBox> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains a specified BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingbox_)>
    fn Contains(&self, box_: &BoundingBox) -> ContainmentType {
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Create 6 planes (do it inline to encourage use of registers)
        let mut NearPlane: XMVECTOR = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        NearPlane = internal::XMPlaneTransform(NearPlane, vOrientation, vOrigin);
        NearPlane = XMPlaneNormalize(NearPlane);

        let mut FarPlane: XMVECTOR = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        FarPlane = internal::XMPlaneTransform(FarPlane, vOrientation, vOrigin);
        FarPlane = XMPlaneNormalize(FarPlane);

        let mut RightPlane: XMVECTOR = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        RightPlane = internal::XMPlaneTransform(RightPlane, vOrientation, vOrigin);
        RightPlane = XMPlaneNormalize(RightPlane);

        let mut LeftPlane: XMVECTOR = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        LeftPlane = internal::XMPlaneTransform(LeftPlane, vOrientation, vOrigin);
        LeftPlane = XMPlaneNormalize(LeftPlane);

        let mut TopPlane: XMVECTOR = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        TopPlane = internal::XMPlaneTransform(TopPlane, vOrientation, vOrigin);
        TopPlane = XMPlaneNormalize(TopPlane);

        let mut BottomPlane: XMVECTOR = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
        BottomPlane = internal::XMPlaneTransform(BottomPlane, vOrientation, vOrigin);
        BottomPlane = XMPlaneNormalize(BottomPlane);

        return box_.ContainedBy(NearPlane, FarPlane, RightPlane, LeftPlane, TopPlane, BottomPlane);
    }
}

impl Contains<&BoundingOrientedBox> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains the specified BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingorientedbox_)>
    fn Contains(&self, box_: &BoundingOrientedBox) -> ContainmentType {
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Create 6 planes (do it inline to encourage use of registers)
        let mut NearPlane: XMVECTOR = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        NearPlane = internal::XMPlaneTransform(NearPlane, vOrientation, vOrigin);
        NearPlane = XMPlaneNormalize(NearPlane);

        let mut FarPlane: XMVECTOR = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        FarPlane = internal::XMPlaneTransform(FarPlane, vOrientation, vOrigin);
        FarPlane = XMPlaneNormalize(FarPlane);

        let mut RightPlane: XMVECTOR = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        RightPlane = internal::XMPlaneTransform(RightPlane, vOrientation, vOrigin);
        RightPlane = XMPlaneNormalize(RightPlane);

        let mut LeftPlane: XMVECTOR = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        LeftPlane = internal::XMPlaneTransform(LeftPlane, vOrientation, vOrigin);
        LeftPlane = XMPlaneNormalize(LeftPlane);

        let mut TopPlane: XMVECTOR = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        TopPlane = internal::XMPlaneTransform(TopPlane, vOrientation, vOrigin);
        TopPlane = XMPlaneNormalize(TopPlane);

        let mut BottomPlane: XMVECTOR = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
        BottomPlane = internal::XMPlaneTransform(BottomPlane, vOrientation, vOrigin);
        BottomPlane = XMPlaneNormalize(BottomPlane);

        return box_.ContainedBy(NearPlane, FarPlane, RightPlane, LeftPlane, TopPlane, BottomPlane);
    }
}

impl Contains<&BoundingFrustum> for BoundingFrustum {
    /// Tests whether the BoundingFrustum contains the specified BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-contains(constboundingfrustum_)>
    fn Contains(&self, fr: &BoundingFrustum) -> ContainmentType {
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // Create 6 planes (do it inline to encourage use of registers)
        let mut NearPlane: XMVECTOR = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        NearPlane = internal::XMPlaneTransform(NearPlane, vOrientation, vOrigin);
        NearPlane = XMPlaneNormalize(NearPlane);

        let mut FarPlane: XMVECTOR = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        FarPlane = internal::XMPlaneTransform(FarPlane, vOrientation, vOrigin);
        FarPlane = XMPlaneNormalize(FarPlane);

        let mut RightPlane: XMVECTOR = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        RightPlane = internal::XMPlaneTransform(RightPlane, vOrientation, vOrigin);
        RightPlane = XMPlaneNormalize(RightPlane);

        let mut LeftPlane: XMVECTOR = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        LeftPlane = internal::XMPlaneTransform(LeftPlane, vOrientation, vOrigin);
        LeftPlane = XMPlaneNormalize(LeftPlane);

        let mut TopPlane: XMVECTOR = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        TopPlane = internal::XMPlaneTransform(TopPlane, vOrientation, vOrigin);
        TopPlane = XMPlaneNormalize(TopPlane);

        let mut BottomPlane: XMVECTOR = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
        BottomPlane = internal::XMPlaneTransform(BottomPlane, vOrientation, vOrigin);
        BottomPlane = XMPlaneNormalize(BottomPlane);

        return fr.ContainedBy(NearPlane, FarPlane, RightPlane, LeftPlane, TopPlane, BottomPlane);
    }
}

//-----------------------------------------------------------------------------
// Exact sphere vs frustum test.  The algorithm first checks the sphere against
// the planes of the frustum, then if the plane checks were indeterminate finds
// the nearest feature (plane, line, point) on the frustum to the center of the
// sphere and compares the distance to the nearest feature to the radius of the
// sphere
//-----------------------------------------------------------------------------
impl Intersects<&BoundingSphere> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a BoundingSphere.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects>
    fn Intersects(&self, sh: &BoundingSphere) -> bool {
        let Zero: XMVECTOR = XMVectorZero();

        // Build the frustum planes.
        let mut Planes: [XMVECTOR; 6] = unsafe { uninitialized() };
        Planes[0] = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        Planes[1] = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        Planes[2] = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        Planes[3] = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        Planes[4] = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        Planes[5] = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
    
        // Normalize the planes so we can compare to the sphere radius.
        Planes[2] = XMVector3Normalize(Planes[2]);
        Planes[3] = XMVector3Normalize(Planes[3]);
        Planes[4] = XMVector3Normalize(Planes[4]);
        Planes[5] = XMVector3Normalize(Planes[5]);
    
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);
    
        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));
    
        // Load the sphere.
        let mut vCenter: XMVECTOR = XMLoadFloat3(&sh.Center);
        let vRadius: XMVECTOR = XMVectorReplicatePtr(&sh.Radius);
    
        // Transform the center of the sphere into the local space of frustum.
        vCenter = XMVector3InverseRotate(XMVectorSubtract(vCenter, vOrigin), vOrientation);
    
        // Set w of the center to one so we can dot4 with the plane.
        vCenter = XMVectorInsert(vCenter, XMVectorSplatOne(), 0, 0, 0, 0, 1);
    
        // Check against each plane of the frustum.
        let mut Outside: XMVECTOR = XMVectorFalseInt();
        let mut InsideAll: XMVECTOR = XMVectorTrueInt();
        let mut CenterInsideAll: XMVECTOR = XMVectorTrueInt();
    
        let mut Dist: [XMVECTOR; 6] = unsafe { uninitialized() };
    
        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            Dist[i] = XMVector4Dot(vCenter, Planes[i]);
    
            // Outside the plane?
            Outside = XMVectorOrInt(Outside, XMVectorGreater(Dist[i], vRadius));
    
            // Fully inside the plane?
            InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(Dist[i], XMVectorNegate(vRadius)));
    
            // Check if the center is inside the plane.
            CenterInsideAll = XMVectorAndInt(CenterInsideAll, XMVectorLessOrEqual(Dist[i], Zero));
        }
    
        // If the sphere is outside any of the planes it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return false;
        }
    
        // If the sphere is inside all planes it is fully inside.
        if (XMVector4EqualInt(InsideAll, XMVectorTrueInt())) {
            return true;
        }
    
        // If the center of the sphere is inside all planes and the sphere intersects
        // one or more planes then it must intersect.
        if (XMVector4EqualInt(CenterInsideAll, XMVectorTrueInt())) {
            return true;
        }
    
        // The sphere may be outside the frustum or intersecting the frustum.
        // Find the nearest feature (face, edge, or corner) on the frustum
        // to the sphere.
    
        // The faces adjacent to each face are:
        //static const size_t adjacent_faces[6][4] =
        const adjacent_faces: [[usize; 4]; 6] = [
            [ 2, 3, 4, 5 ],    // 0
            [ 2, 3, 4, 5 ],    // 1
            [ 0, 1, 4, 5 ],    // 2
            [ 0, 1, 4, 5 ],    // 3
            [ 0, 1, 2, 3 ],    // 4
            [ 0, 1, 2, 3 ]
        ];  // 5
    
        let mut Intersects: XMVECTOR = XMVectorFalseInt();
    
        // Check to see if the nearest feature is one of the planes.
        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            // Find the nearest point on the plane to the center of the sphere.
            let mut Point: XMVECTOR = XMVectorNegativeMultiplySubtract(Planes[i], Dist[i], vCenter);
    
            // Set w of the point to one.
            Point = XMVectorInsert(Point, XMVectorSplatOne(), 0, 0, 0, 0, 1);
    
            // If the point is inside the face (inside the adjacent planes) then
            // this plane is the nearest feature.
            let mut InsideFace: XMVECTOR = XMVectorTrueInt();
    
            //for (let j: size_t = 0; j < 4; j++)
            for j in 0 .. 4
            {
                let plane_index: usize = adjacent_faces[i][j];
    
                InsideFace = XMVectorAndInt(InsideFace,
                    XMVectorLessOrEqual(XMVector4Dot(Point, Planes[plane_index]), Zero));
            }
    
            // Since we have already checked distance from the plane we know that the
            // sphere must intersect if this plane is the nearest feature.
            Intersects = XMVectorOrInt(Intersects,
                XMVectorAndInt(XMVectorGreater(Dist[i], Zero), InsideFace));
        }
    
        if (XMVector4EqualInt(Intersects, XMVectorTrueInt())) {
            return true;
        }
    
        // Build the corners of the frustum.
        let vRightTop: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
        let vRightBottom: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
        let vLeftTop: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
        let vLeftBottom: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
        let vNear: XMVECTOR = XMVectorReplicatePtr(&self.Near);
        let vFar: XMVECTOR = XMVectorReplicatePtr(&self.Far);
    
        let mut Corners: [XMVECTOR; BoundingFrustum::CORNER_COUNT] = unsafe { uninitialized() };
        Corners[0] = XMVectorMultiply(vRightTop, vNear);
        Corners[1] = XMVectorMultiply(vRightBottom, vNear);
        Corners[2] = XMVectorMultiply(vLeftTop, vNear);
        Corners[3] = XMVectorMultiply(vLeftBottom, vNear);
        Corners[4] = XMVectorMultiply(vRightTop, vFar);
        Corners[5] = XMVectorMultiply(vRightBottom, vFar);
        Corners[6] = XMVectorMultiply(vLeftTop, vFar);
        Corners[7] = XMVectorMultiply(vLeftBottom, vFar);
    
        // The Edges are:
        //static const size_t edges[12][2] =
        const edges: [[usize; 2]; 12] =
        [
            [ 0, 1 ], [ 2, 3 ], [ 0, 2 ], [ 1, 3 ],    // Near plane
            [ 4, 5 ], [ 6, 7 ], [ 4, 6 ], [ 5, 7 ],    // Far plane
            [ 0, 4 ], [ 1, 5 ], [ 2, 6 ], [ 3, 7 ],
        ]; // Near to far
    
        let RadiusSq: XMVECTOR = XMVectorMultiply(vRadius, vRadius);
    
        // Check to see if the nearest feature is one of the edges (or corners).
        //for (let i: size_t = 0; i < 12; ++i)
        for i in 0 .. 12
        {
            let ei0: usize = edges[i][0];
            let ei1: usize = edges[i][1];
    
            // Find the nearest point on the edge to the center of the sphere.
            // The corners of the frustum are included as the endpoints of the edges.
            let Point: XMVECTOR = internal::PointOnLineSegmentNearestPoint(Corners[ei0], Corners[ei1], vCenter);
    
            let Delta: XMVECTOR = XMVectorSubtract(vCenter, Point);
    
            let DistSq: XMVECTOR = XMVector3Dot(Delta, Delta);
    
            // If the distance to the center of the sphere to the point is less than
            // the radius of the sphere then it must intersect.
            Intersects = XMVectorOrInt(Intersects, XMVectorLessOrEqual(DistSq, RadiusSq));
        }
    
        if (XMVector4EqualInt(Intersects, XMVectorTrueInt())) {
            return true;
        }
    
        // The sphere must be outside the frustum.
        return false;        
    }
}

impl Intersects<&BoundingBox> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a BoundingBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(constboundingbox_)>
    fn Intersects(&self, box_: &BoundingBox) -> bool {
        // Make the axis aligned box oriented and do an OBB vs frustum test.
        let obox = BoundingOrientedBox {
            Center: box_.Center,
            Extents: box_.Extents,
            Orientation: XMFLOAT4::set(0.0, 0.0, 0.0, 1.0)
        };
        return self.Intersects(&obox);
    }
}

impl Intersects<&BoundingOrientedBox> for BoundingFrustum {
    /// Test the BoundingFrustum for intersection with a BoundingOrientedBox.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(constboundingorientedbox_)>
    fn Intersects(&self, box_: &BoundingOrientedBox) -> bool {
        const SelectY: XMVECTOR = unsafe { XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_1, XM_SELECT_0, XM_SELECT_0 ] }.v };
        const SelectZ: XMVECTOR = unsafe { XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_0, XM_SELECT_1, XM_SELECT_0 ] }.v };
    
        let Zero: XMVECTOR = XMVectorZero();
    
        // Build the frustum planes.
        let mut Planes: [XMVECTOR; 6] = unsafe { uninitialized() };
        Planes[0] = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        Planes[1] = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        Planes[2] = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        Planes[3] = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        Planes[4] = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        Planes[5] = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
    
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let FrustumOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);
    
        debug_assert!(internal::XMQuaternionIsUnit(FrustumOrientation));
    
        // Load the box.
        let mut Center: XMVECTOR = XMLoadFloat3(&box_.Center);
        let Extents: XMVECTOR = XMLoadFloat3(&box_.Extents);
        let mut BoxOrientation: XMVECTOR = XMLoadFloat4(&box_.Orientation);
    
        debug_assert!(internal::XMQuaternionIsUnit(BoxOrientation));
    
        // Transform the oriented box into the space of the frustum in order to
        // minimize the number of transforms we have to do.
        Center = XMVector3InverseRotate(XMVectorSubtract(Center, vOrigin), FrustumOrientation);
        BoxOrientation = XMQuaternionMultiply(BoxOrientation, XMQuaternionConjugate(FrustumOrientation));
    
        // Set w of the center to one so we can dot4 with the plane.
        Center = XMVectorInsert(Center, XMVectorSplatOne(), 0, 0, 0, 0, 1);
    
        // Build the 3x3 rotation matrix that defines the box axes.
        let R: XMMATRIX = XMMatrixRotationQuaternion(BoxOrientation);
    
        // Check against each plane of the frustum.
        let mut Outside: XMVECTOR = XMVectorFalseInt();
        let mut InsideAll: XMVECTOR = XMVectorTrueInt();
        let mut CenterInsideAll: XMVECTOR = XMVectorTrueInt();
    
        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            // Compute the distance to the center of the box.
            let Dist: XMVECTOR = XMVector4Dot(Center, Planes[i]);
    
            // Project the axes of the box onto the normal of the plane.  Half the
            // length of the projection (sometime called the "radius") is equal to
            // h(u) * abs(n dot b(u))) + h(v) * abs(n dot b(v)) + h(w) * abs(n dot b(w))
            // where h(i) are extents of the box, n is the plane normal, and b(i) are the
            // axes of the box.
            unsafe {
                let mut Radius: XMVECTOR = XMVector3Dot(Planes[i], R.r[0]);
                Radius = XMVectorSelect(Radius, XMVector3Dot(Planes[i], R.r[1]), SelectY);
                Radius = XMVectorSelect(Radius, XMVector3Dot(Planes[i], R.r[2]), SelectZ);
                Radius = XMVector3Dot(Extents, XMVectorAbs(Radius));
        
                // Outside the plane?
                Outside = XMVectorOrInt(Outside, XMVectorGreater(Dist, Radius));
        
                // Fully inside the plane?
                InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(Dist, XMVectorNegate(Radius)));
        
                // Check if the center is inside the plane.
                CenterInsideAll = XMVectorAndInt(CenterInsideAll, XMVectorLessOrEqual(Dist, Zero));
            }
        }
    
        // If the box is outside any of the planes it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return false;
        }
    
        // If the box is inside all planes it is fully inside.
        if (XMVector4EqualInt(InsideAll, XMVectorTrueInt())) {
            return true;
        }
    
        // If the center of the box is inside all planes and the box intersects
        // one or more planes then it must intersect.
        if (XMVector4EqualInt(CenterInsideAll, XMVectorTrueInt())) {
            return true;
        }
    
        // Build the corners of the frustum.
        let vRightTop: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
        let vRightBottom: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
        let vLeftTop: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
        let vLeftBottom: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
        let vNear: XMVECTOR = XMVectorReplicatePtr(&self.Near);
        let vFar: XMVECTOR = XMVectorReplicatePtr(&self.Far);
    
        let mut Corners: [XMVECTOR; Self::CORNER_COUNT] = unsafe { uninitialized() };
        Corners[0] = XMVectorMultiply(vRightTop, vNear);
        Corners[1] = XMVectorMultiply(vRightBottom, vNear);
        Corners[2] = XMVectorMultiply(vLeftTop, vNear);
        Corners[3] = XMVectorMultiply(vLeftBottom, vNear);
        Corners[4] = XMVectorMultiply(vRightTop, vFar);
        Corners[5] = XMVectorMultiply(vRightBottom, vFar);
        Corners[6] = XMVectorMultiply(vLeftTop, vFar);
        Corners[7] = XMVectorMultiply(vLeftBottom, vFar);
    
        unsafe {
            // Test against box axes (3)
            {
                // Find the min/max values of the projection of the frustum onto each axis.
                let mut FrustumMin: XMVECTOR;
                let mut FrustumMax: XMVECTOR;
        
                FrustumMin = XMVector3Dot(Corners[0], R.r[0]);
                FrustumMin = XMVectorSelect(FrustumMin, XMVector3Dot(Corners[0], R.r[1]), SelectY);
                FrustumMin = XMVectorSelect(FrustumMin, XMVector3Dot(Corners[0], R.r[2]), SelectZ);
                FrustumMax = FrustumMin;
        
                //for (let i: size_t = 1; i < BoundingOrientedBox::CORNER_COUNT; ++i)
                for i in 1 .. BoundingOrientedBox::CORNER_COUNT
                {
                    let mut Temp: XMVECTOR = XMVector3Dot(Corners[i], R.r[0]);
                    Temp = XMVectorSelect(Temp, XMVector3Dot(Corners[i], R.r[1]), SelectY);
                    Temp = XMVectorSelect(Temp, XMVector3Dot(Corners[i], R.r[2]), SelectZ);
        
                    FrustumMin = XMVectorMin(FrustumMin, Temp);
                    FrustumMax = XMVectorMax(FrustumMax, Temp);
                }
        
                // Project the center of the box onto the axes.
                let mut BoxDist: XMVECTOR = XMVector3Dot(Center, R.r[0]);
                BoxDist = XMVectorSelect(BoxDist, XMVector3Dot(Center, R.r[1]), SelectY);
                BoxDist = XMVectorSelect(BoxDist, XMVector3Dot(Center, R.r[2]), SelectZ);
        
                // The projection of the box onto the axis is just its Center and Extents.
                // if (min > box_max || max < box_min) reject;
                let Result: XMVECTOR = XMVectorOrInt(XMVectorGreater(FrustumMin, XMVectorAdd(BoxDist, Extents)),
                    XMVectorLess(FrustumMax, XMVectorSubtract(BoxDist, Extents)));
        
                if (internal::XMVector3AnyTrue(Result)) {
                    return false;
                }
            }
        
            // Test against edge/edge axes (3*6).
            let mut FrustumEdgeAxis: [XMVECTOR; 6] = uninitialized();
        
            FrustumEdgeAxis[0] = vRightTop;
            FrustumEdgeAxis[1] = vRightBottom;
            FrustumEdgeAxis[2] = vLeftTop;
            FrustumEdgeAxis[3] = vLeftBottom;
            FrustumEdgeAxis[4] = XMVectorSubtract(vRightTop, vLeftTop);
            FrustumEdgeAxis[5] = XMVectorSubtract(vLeftBottom, vLeftTop);
        
            //for (let i: size_t = 0; i < 3; ++i)
            for i in 0 .. 3
            {
                //for (let j: size_t = 0; j < 6; j++)
                for j in 0 .. 6
                {
                    // Compute the axis we are going to test.
                    let Axis: XMVECTOR = XMVector3Cross(R.r[i], FrustumEdgeAxis[j]);
        
                    // Find the min/max values of the projection of the frustum onto the axis.
                    let mut FrustumMin: XMVECTOR;
                    let mut FrustumMax: XMVECTOR;
        
                    FrustumMin = XMVector3Dot(Axis, Corners[0]);
                    FrustumMax = FrustumMin;
        
                    //for (let k: size_t = 1; k < CORNER_COUNT; k++)
                    for k in 1 .. Self::CORNER_COUNT
                    {
                        let Temp: XMVECTOR = XMVector3Dot(Axis, Corners[k]);
                        FrustumMin = XMVectorMin(FrustumMin, Temp);
                        FrustumMax = XMVectorMax(FrustumMax, Temp);
                    }
        
                    // Project the center of the box onto the axis.
                    let Dist: XMVECTOR = XMVector3Dot(Center, Axis);
        
                    // Project the axes of the box onto the axis to find the "radius" of the box.
                    let mut Radius: XMVECTOR = XMVector3Dot(Axis, R.r[0]);
                    Radius = XMVectorSelect(Radius, XMVector3Dot(Axis, R.r[1]), SelectY);
                    Radius = XMVectorSelect(Radius, XMVector3Dot(Axis, R.r[2]), SelectZ);
                    Radius = XMVector3Dot(Extents, XMVectorAbs(Radius));
        
                    // if (center > max + radius || center < min - radius) reject;
                    Outside = XMVectorOrInt(Outside, XMVectorGreater(Dist, XMVectorAdd(FrustumMax, Radius)));
                    Outside = XMVectorOrInt(Outside, XMVectorLess(Dist, XMVectorSubtract(FrustumMin, Radius)));
                }
            }
        }
    
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return false;
        }
    
        // If we did not find a separating plane then the box must intersect the frustum.
        return true;
    }
}

impl Intersects<&BoundingFrustum> for BoundingFrustum {
    /// Test the BoundingFrustum for intersection with a BoundingFrustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(constboundingfrustum_)>
    fn Intersects(&self, fr: &BoundingFrustum) -> bool {
        // Load origin and orientation of frustum B.
        let OriginB: XMVECTOR = XMLoadFloat3(&self.Origin);
        let OrientationB: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(OrientationB));

        // Build the planes of frustum B.
        let mut AxisB: [XMVECTOR; 6] = unsafe { uninitialized() };
        AxisB[0] = XMVectorSet(0.0, 0.0, -1.0, 0.0);
        AxisB[1] = XMVectorSet(0.0, 0.0, 1.0, 0.0);
        AxisB[2] = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        AxisB[3] = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        AxisB[4] = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        AxisB[5] = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);

        let mut PlaneDistB: [XMVECTOR; 6] = unsafe { uninitialized() };
        PlaneDistB[0] = XMVectorNegate(XMVectorReplicatePtr(&self.Near));
        PlaneDistB[1] = XMVectorReplicatePtr(&self.Far);
        PlaneDistB[2] = XMVectorZero();
        PlaneDistB[3] = XMVectorZero();
        PlaneDistB[4] = XMVectorZero();
        PlaneDistB[5] = XMVectorZero();

        // Load origin and orientation of frustum A.
        let mut OriginA: XMVECTOR = XMLoadFloat3(&fr.Origin);
        let mut OrientationA: XMVECTOR = XMLoadFloat4(&fr.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(OrientationA));

        // Transform frustum A into the space of the frustum B in order to
        // minimize the number of transforms we have to do.
        OriginA = XMVector3InverseRotate(XMVectorSubtract(OriginA, OriginB), OrientationB);
        OrientationA = XMQuaternionMultiply(OrientationA, XMQuaternionConjugate(OrientationB));

        // Build the corners of frustum A (in the local space of B).
        let mut RightTopA: XMVECTOR = XMVectorSet(fr.RightSlope, fr.TopSlope, 1.0, 0.0);
        let mut RightBottomA: XMVECTOR = XMVectorSet(fr.RightSlope, fr.BottomSlope, 1.0, 0.0);
        let mut LeftTopA: XMVECTOR = XMVectorSet(fr.LeftSlope, fr.TopSlope, 1.0, 0.0);
        let mut LeftBottomA: XMVECTOR = XMVectorSet(fr.LeftSlope, fr.BottomSlope, 1.0, 0.0);
        let NearA: XMVECTOR = XMVectorReplicatePtr(&fr.Near);
        let FarA: XMVECTOR = XMVectorReplicatePtr(&fr.Far);

        RightTopA = XMVector3Rotate(RightTopA, OrientationA);
        RightBottomA = XMVector3Rotate(RightBottomA, OrientationA);
        LeftTopA = XMVector3Rotate(LeftTopA, OrientationA);
        LeftBottomA = XMVector3Rotate(LeftBottomA, OrientationA);

        let mut CornersA: [XMVECTOR; Self::CORNER_COUNT] = unsafe { uninitialized() };
        CornersA[0] = XMVectorMultiplyAdd(RightTopA, NearA, OriginA);
        CornersA[1] = XMVectorMultiplyAdd(RightBottomA, NearA, OriginA);
        CornersA[2] = XMVectorMultiplyAdd(LeftTopA, NearA, OriginA);
        CornersA[3] = XMVectorMultiplyAdd(LeftBottomA, NearA, OriginA);
        CornersA[4] = XMVectorMultiplyAdd(RightTopA, FarA, OriginA);
        CornersA[5] = XMVectorMultiplyAdd(RightBottomA, FarA, OriginA);
        CornersA[6] = XMVectorMultiplyAdd(LeftTopA, FarA, OriginA);
        CornersA[7] = XMVectorMultiplyAdd(LeftBottomA, FarA, OriginA);

        // Check frustum A against each plane of frustum B.
        let mut Outside: XMVECTOR = XMVectorFalseInt();
        let mut InsideAll: XMVECTOR = XMVectorTrueInt();

        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            // Find the min/max projection of the frustum onto the plane normal.
            let mut Min: XMVECTOR;
            let mut Max: XMVECTOR;

            Min = XMVector3Dot(AxisB[i], CornersA[0]);
            Max = Min;

            //for (let j: size_t = 1; j < CORNER_COUNT; j++)
            for j in 1 .. Self::CORNER_COUNT
            {
                let Temp: XMVECTOR = XMVector3Dot(AxisB[i], CornersA[j]);
                Min = XMVectorMin(Min, Temp);
                Max = XMVectorMax(Max, Temp);
            }

            // Outside the plane?
            Outside = XMVectorOrInt(Outside, XMVectorGreater(Min, PlaneDistB[i]));

            // Fully inside the plane?
            InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(Max, PlaneDistB[i]));
        }

        // If the frustum A is outside any of the planes of frustum B it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return false;
        }

        // If frustum A is inside all planes of frustum B it is fully inside.
        if (XMVector4EqualInt(InsideAll, XMVectorTrueInt())) {
            return true;
        }

        // Build the corners of frustum B.
        let RightTopB: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
        let RightBottomB: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
        let LeftTopB: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
        let LeftBottomB: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
        let NearB: XMVECTOR = XMVectorReplicatePtr(&self.Near);
        let FarB: XMVECTOR = XMVectorReplicatePtr(&self.Far);

        let mut CornersB: [XMVECTOR; BoundingFrustum::CORNER_COUNT] = unsafe { uninitialized() };
        CornersB[0] = XMVectorMultiply(RightTopB, NearB);
        CornersB[1] = XMVectorMultiply(RightBottomB, NearB);
        CornersB[2] = XMVectorMultiply(LeftTopB, NearB);
        CornersB[3] = XMVectorMultiply(LeftBottomB, NearB);
        CornersB[4] = XMVectorMultiply(RightTopB, FarB);
        CornersB[5] = XMVectorMultiply(RightBottomB, FarB);
        CornersB[6] = XMVectorMultiply(LeftTopB, FarB);
        CornersB[7] = XMVectorMultiply(LeftBottomB, FarB);

        // Build the planes of frustum A (in the local space of B).
        let mut AxisA: [XMVECTOR; 6] = unsafe { uninitialized() };
        let mut PlaneDistA: [XMVECTOR; 6] = unsafe { uninitialized() };

        AxisA[0] = XMVectorSet(0.0, 0.0, -1.0, 0.0);
        AxisA[1] = XMVectorSet(0.0, 0.0, 1.0, 0.0);
        AxisA[2] = XMVectorSet(1.0, 0.0, -fr.RightSlope, 0.0);
        AxisA[3] = XMVectorSet(-1.0, 0.0, fr.LeftSlope, 0.0);
        AxisA[4] = XMVectorSet(0.0, 1.0, -fr.TopSlope, 0.0);
        AxisA[5] = XMVectorSet(0.0, -1.0, fr.BottomSlope, 0.0);

        AxisA[0] = XMVector3Rotate(AxisA[0], OrientationA);
        AxisA[1] = XMVectorNegate(AxisA[0]);
        AxisA[2] = XMVector3Rotate(AxisA[2], OrientationA);
        AxisA[3] = XMVector3Rotate(AxisA[3], OrientationA);
        AxisA[4] = XMVector3Rotate(AxisA[4], OrientationA);
        AxisA[5] = XMVector3Rotate(AxisA[5], OrientationA);

        PlaneDistA[0] = XMVector3Dot(AxisA[0], CornersA[0]);  // Re-use corner on near plane.
        PlaneDistA[1] = XMVector3Dot(AxisA[1], CornersA[4]);  // Re-use corner on far plane.
        PlaneDistA[2] = XMVector3Dot(AxisA[2], OriginA);
        PlaneDistA[3] = XMVector3Dot(AxisA[3], OriginA);
        PlaneDistA[4] = XMVector3Dot(AxisA[4], OriginA);
        PlaneDistA[5] = XMVector3Dot(AxisA[5], OriginA);

        // Check each axis of frustum A for a seperating plane (5).
        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            // Find the minimum projection of the frustum onto the plane normal.
            let mut Min: XMVECTOR;

            Min = XMVector3Dot(AxisA[i], CornersB[0]);

            //for (let j: size_t = 1; j < CORNER_COUNT; j++)
            for j in 0 .. Self::CORNER_COUNT
            {
                let Temp: XMVECTOR = XMVector3Dot(AxisA[i], CornersB[j]);
                Min = XMVectorMin(Min, Temp);
            }

            // Outside the plane?
            Outside = XMVectorOrInt(Outside, XMVectorGreater(Min, PlaneDistA[i]));
        }

        // If the frustum B is outside any of the planes of frustum A it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return false;
        }

        // Check edge/edge axes (6 * 6).
        let mut FrustumEdgeAxisA: [XMVECTOR; 6] = unsafe { uninitialized() };
        FrustumEdgeAxisA[0] = RightTopA;
        FrustumEdgeAxisA[1] = RightBottomA;
        FrustumEdgeAxisA[2] = LeftTopA;
        FrustumEdgeAxisA[3] = LeftBottomA;
        FrustumEdgeAxisA[4] = XMVectorSubtract(RightTopA, LeftTopA);
        FrustumEdgeAxisA[5] = XMVectorSubtract(LeftBottomA, LeftTopA);

        let mut FrustumEdgeAxisB: [XMVECTOR; 6] = unsafe { uninitialized() };
        FrustumEdgeAxisB[0] = RightTopB;
        FrustumEdgeAxisB[1] = RightBottomB;
        FrustumEdgeAxisB[2] = LeftTopB;
        FrustumEdgeAxisB[3] = LeftBottomB;
        FrustumEdgeAxisB[4] = XMVectorSubtract(RightTopB, LeftTopB);
        FrustumEdgeAxisB[5] = XMVectorSubtract(LeftBottomB, LeftTopB);

        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            //for (let j: size_t = 0; j < 6; j++)
            for j in 0 .. 6
            {
                // Compute the axis we are going to test.
                let Axis: XMVECTOR = XMVector3Cross(FrustumEdgeAxisA[i], FrustumEdgeAxisB[j]);

                // Find the min/max values of the projection of both frustums onto the axis.
                let mut MinA: XMVECTOR; let mut MaxA: XMVECTOR;
                let mut MinB: XMVECTOR; let mut MaxB: XMVECTOR;

                MinA = XMVector3Dot(Axis, CornersA[0]);
                MaxA = MinA;
                MinB = XMVector3Dot(Axis, CornersB[0]);
                MaxB = MinB;

                //for (let k: size_t = 1; k < CORNER_COUNT; k++)
                for k in 1 .. Self::CORNER_COUNT
                {
                    let TempA: XMVECTOR = XMVector3Dot(Axis, CornersA[k]);
                    MinA = XMVectorMin(MinA, TempA);
                    MaxA = XMVectorMax(MaxA, TempA);

                    let TempB: XMVECTOR = XMVector3Dot(Axis, CornersB[k]);
                    MinB = XMVectorMin(MinB, TempB);
                    MaxB = XMVectorMax(MaxB, TempB);
                }

                // if (MinA > MaxB || MinB > MaxA) reject
                Outside = XMVectorOrInt(Outside, XMVectorGreater(MinA, MaxB));
                Outside = XMVectorOrInt(Outside, XMVectorGreater(MinB, MaxA));
            }
        }

        // If there is a seperating plane, then the frustums do not intersect.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return false;
        }

        // If we did not find a separating plane then the frustums intersect.
        return true;
    }
}

impl Intersects<Triangle> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a triangle.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(fxmvector_fxmvector_fxmvector)>
    fn Intersects(&self, (V0, V1, V2): Triangle) -> bool {
       // Build the frustum planes (NOTE: D is negated from the usual).
       let mut Planes: [XMVECTOR; 6] = unsafe { uninitialized() };
       Planes[0] = XMVectorSet(0.0, 0.0, -1.0, -self.Near);
       Planes[1] = XMVectorSet(0.0, 0.0, 1.0, self.Far);
       Planes[2] = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
       Planes[3] = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
       Planes[4] = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
       Planes[5] = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);

       // Load origin and orientation of the frustum.
       let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
       let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

       debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

       // Transform triangle into the local space of frustum.
       let TV0: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V0, vOrigin), vOrientation);
       let TV1: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V1, vOrigin), vOrientation);
       let TV2: XMVECTOR = XMVector3InverseRotate(XMVectorSubtract(V2, vOrigin), vOrientation);

       // Test each vertex of the triangle against the frustum planes.
       let mut Outside: XMVECTOR = XMVectorFalseInt();
       let mut InsideAll: XMVECTOR = XMVectorTrueInt();

       for i in 0 .. 6
       {
           let Dist0: XMVECTOR = XMVector3Dot(TV0, Planes[i]);
           let Dist1: XMVECTOR = XMVector3Dot(TV1, Planes[i]);
           let Dist2: XMVECTOR = XMVector3Dot(TV2, Planes[i]);

           let mut MinDist: XMVECTOR = XMVectorMin(Dist0, Dist1);
           MinDist = XMVectorMin(MinDist, Dist2);
           let mut MaxDist: XMVECTOR = XMVectorMax(Dist0, Dist1);
           MaxDist = XMVectorMax(MaxDist, Dist2);

           let PlaneDist: XMVECTOR = XMVectorSplatW(Planes[i]);

           // Outside the plane?
           Outside = XMVectorOrInt(Outside, XMVectorGreater(MinDist, PlaneDist));

           // Fully inside the plane?
           InsideAll = XMVectorAndInt(InsideAll, XMVectorLessOrEqual(MaxDist, PlaneDist));
       }

       // If the triangle is outside any of the planes it is outside.
       if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
           return false;
       }

       // If the triangle is inside all planes it is fully inside.
       if (XMVector4EqualInt(InsideAll, XMVectorTrueInt())) {
           return true;
       }

       // Build the corners of the frustum.
       let vRightTop: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
       let vRightBottom: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
       let vLeftTop: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
       let vLeftBottom: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
       let vNear: XMVECTOR = XMVectorReplicatePtr(&self.Near);
       let vFar: XMVECTOR = XMVectorReplicatePtr(&self.Far);

       let mut Corners: [XMVECTOR; Self::CORNER_COUNT] = unsafe { uninitialized() };
       Corners[0] = XMVectorMultiply(vRightTop, vNear);
       Corners[1] = XMVectorMultiply(vRightBottom, vNear);
       Corners[2] = XMVectorMultiply(vLeftTop, vNear);
       Corners[3] = XMVectorMultiply(vLeftBottom, vNear);
       Corners[4] = XMVectorMultiply(vRightTop, vFar);
       Corners[5] = XMVectorMultiply(vRightBottom, vFar);
       Corners[6] = XMVectorMultiply(vLeftTop, vFar);
       Corners[7] = XMVectorMultiply(vLeftBottom, vFar);

       // Test the plane of the triangle.
       let Normal: XMVECTOR = XMVector3Cross(XMVectorSubtract(V1, V0), XMVectorSubtract(V2, V0));
       let Dist: XMVECTOR = XMVector3Dot(Normal, V0);

       let mut MinDist: XMVECTOR;
       let mut MaxDist: XMVECTOR;

       MinDist = XMVector3Dot(Corners[0], Normal);
       MaxDist = MinDist;

       for i in 1 .. Self::CORNER_COUNT
       {
           let Temp: XMVECTOR = XMVector3Dot(Corners[i], Normal);
           MinDist = XMVectorMin(MinDist, Temp);
           MaxDist = XMVectorMax(MaxDist, Temp);
       }

       Outside = XMVectorOrInt(XMVectorGreater(MinDist, Dist), XMVectorLess(MaxDist, Dist));
       if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
           return false;
       }

       // Check the edge/edge axes (3*6).
       let mut TriangleEdgeAxis: [XMVECTOR; 3] = unsafe { uninitialized() };
       TriangleEdgeAxis[0] = XMVectorSubtract(V1, V0);
       TriangleEdgeAxis[1] = XMVectorSubtract(V2, V1);
       TriangleEdgeAxis[2] = XMVectorSubtract(V0, V2);

       let mut FrustumEdgeAxis: [XMVECTOR; 6] = unsafe { uninitialized() };
       FrustumEdgeAxis[0] = vRightTop;
       FrustumEdgeAxis[1] = vRightBottom;
       FrustumEdgeAxis[2] = vLeftTop;
       FrustumEdgeAxis[3] = vLeftBottom;
       FrustumEdgeAxis[4] = XMVectorSubtract(vRightTop, vLeftTop);
       FrustumEdgeAxis[5] = XMVectorSubtract(vLeftBottom, vLeftTop);

       for i in 0 .. 3
       {
           for j in 0 .. 6
           {
               // Compute the axis we are going to test.
               let Axis: XMVECTOR = XMVector3Cross(TriangleEdgeAxis[i], FrustumEdgeAxis[j]);

               // Find the min/max of the projection of the triangle onto the axis.
               let mut MinA: XMVECTOR;
               let mut MaxA: XMVECTOR;

               let Dist0: XMVECTOR = XMVector3Dot(V0, Axis);
               let Dist1: XMVECTOR = XMVector3Dot(V1, Axis);
               let Dist2: XMVECTOR = XMVector3Dot(V2, Axis);

               MinA = XMVectorMin(Dist0, Dist1);
               MinA = XMVectorMin(MinA, Dist2);
               MaxA = XMVectorMax(Dist0, Dist1);
               MaxA = XMVectorMax(MaxA, Dist2);

               // Find the min/max of the projection of the frustum onto the axis.
               let mut MinB: XMVECTOR;
               let mut MaxB: XMVECTOR;

               MinB = XMVector3Dot(Axis, Corners[0]);
               MaxB = MinB;

               //for (let k: size_t = 1; k < CORNER_COUNT; k++)
               for k in 1 .. Self::CORNER_COUNT
               {
                   let Temp: XMVECTOR = XMVector3Dot(Axis, Corners[k]);
                   MinB = XMVectorMin(MinB, Temp);
                   MaxB = XMVectorMax(MaxB, Temp);
               }

               // if (MinA > MaxB || MinB > MaxA) reject;
               Outside = XMVectorOrInt(Outside, XMVectorGreater(MinA, MaxB));
               Outside = XMVectorOrInt(Outside, XMVectorGreater(MinB, MaxA));
           }
       }

       if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
           return false;
       }

       // If we did not find a separating plane then the triangle must intersect the frustum.
       return true;
    }
}

impl Intersects<Plane, PlaneIntersectionType> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a Plane.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(fxmvector)>
    fn Intersects(&self, Plane: Plane) -> PlaneIntersectionType {
        debug_assert!(internal::XMPlaneIsUnit(Plane));

        // Load origin and orientation of the frustum.
        let mut vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);
    
        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));
    
        // Set w of the origin to one so we can dot4 with a plane.
        // TODO: template
        vOrigin = XMVectorInsert(vOrigin, XMVectorSplatOne(), 0, 0, 0, 0, 1);
    
        // Build the corners of the frustum (in world space).
        let mut RightTop: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
        let mut RightBottom: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
        let mut LeftTop: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
        let mut LeftBottom: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
        let vNear: XMVECTOR = XMVectorReplicatePtr(&self.Near);
        let vFar: XMVECTOR = XMVectorReplicatePtr(&self.Far);
    
        RightTop = XMVector3Rotate(RightTop, vOrientation);
        RightBottom = XMVector3Rotate(RightBottom, vOrientation);
        LeftTop = XMVector3Rotate(LeftTop, vOrientation);
        LeftBottom = XMVector3Rotate(LeftBottom, vOrientation);
    
        let Corners0: XMVECTOR = XMVectorMultiplyAdd(RightTop, vNear, vOrigin);
        let Corners1: XMVECTOR = XMVectorMultiplyAdd(RightBottom, vNear, vOrigin);
        let Corners2: XMVECTOR = XMVectorMultiplyAdd(LeftTop, vNear, vOrigin);
        let Corners3: XMVECTOR = XMVectorMultiplyAdd(LeftBottom, vNear, vOrigin);
        let Corners4: XMVECTOR = XMVectorMultiplyAdd(RightTop, vFar, vOrigin);
        let Corners5: XMVECTOR = XMVectorMultiplyAdd(RightBottom, vFar, vOrigin);
        let Corners6: XMVECTOR = XMVectorMultiplyAdd(LeftTop, vFar, vOrigin);
        let Corners7: XMVECTOR = XMVectorMultiplyAdd(LeftBottom, vFar, vOrigin);
    
        let mut Inside: XMVECTOR = unsafe { uninitialized() };
        let mut Outside: XMVECTOR = unsafe { uninitialized() };

        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane, &mut Outside, &mut Inside);
    
        // If the frustum is outside any plane it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return FRONT;
        }
    
        // If the frustum is inside all planes it is inside.
        if (XMVector4EqualInt(Inside, XMVectorTrueInt())) {
            return BACK;
        }
    
        // The frustum is not inside all planes or outside a plane it intersects.
        return INTERSECTING;        
    }
}

impl Intersects<RayMut<'_>> for BoundingFrustum {
    /// Tests the BoundingFrustum for intersection with a ray.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-intersects(fxmvector_fxmvector_float_)>
    fn Intersects(&self, (rayOrigin, Direction, Dist): RayMut) -> bool {
        // If ray starts inside the frustum, return a distance of 0 for the hit
        if (self.Contains(rayOrigin) == CONTAINS)
        {
            *Dist = 0.0;
            return true;
        }

        // Build the frustum planes.
        let mut Planes: [XMVECTOR; 6] = unsafe { uninitialized() };
        Planes[0] = XMVectorSet(0.0, 0.0, -1.0, self.Near);
        Planes[1] = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
        Planes[2] = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
        Planes[3] = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
        Planes[4] = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
        Planes[5] = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);

        // Load origin and orientation of the frustum.
        let frOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let frOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        // This algorithm based on "Fast Ray-Convex Polyhedron Intersectin," in James Arvo, ed., Graphics Gems II pp. 247-250
        let mut tnear: f32 = -FLT_MAX;
        let mut tfar: f32 = FLT_MAX;

        //for (let i: size_t = 0; i < 6; ++i)
        for i in 0 .. 6
        {
            let mut Plane: XMVECTOR = internal::XMPlaneTransform(Planes[i], frOrientation, frOrigin);
            Plane = XMPlaneNormalize(Plane);

            let AxisDotOrigin: XMVECTOR = XMPlaneDotCoord(Plane, rayOrigin);
            let AxisDotDirection: XMVECTOR = XMVector3Dot(Plane, Direction);

            if (XMVector3LessOrEqual(XMVectorAbs(AxisDotDirection), g_RayEpsilon.v()))
            {
                // Ray is parallel to plane - check if ray origin is inside plane's
                if (XMVector3Greater(AxisDotOrigin, g_XMZero.v()))
                {
                    // Ray origin is outside half-space.
                    *Dist = 0.0;
                    return false;
                }
            }
            else
            {
                // Ray not parallel - get distance to plane.
                let vd: f32 = XMVectorGetX(AxisDotDirection);
                let vn: f32 = XMVectorGetX(AxisDotOrigin);
                let t: f32 = -vn / vd;
                if (vd < 0.0)
                {
                    // Front face - T is a near point.
                    if (t > tfar)
                    {
                        *Dist = 0.0;
                        return false;
                    }
                    if (t > tnear)
                    {
                        // Hit near face.
                        tnear = t;
                    }
                }
                else
                {
                    // back face - T is far point.
                    if (t < tnear)
                    {
                        *Dist = 0.0;
                        return false;
                    }
                    if (t < tfar)
                    {
                        // Hit far face.
                        tfar = t;
                    }
                }
            }
        }

        // Survived all tests.
        // Note: if ray originates on polyhedron, may want to change 0.0f to some
        // epsilon to avoid intersecting the originating face.
        let distance: f32 = if (tnear >= 0.00) { tnear } else { tfar };
        if (distance >= 0.0)
        {
            *Dist = distance;
            return true;
        }

        *Dist = 0.0;
        return false;
    }
}

impl ContainedBy for BoundingFrustum {
    /// Tests whether the BoundingFrustum is contained by the specified frustum.
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/api/directxcollision/nf-directxcollision-BoundingFrustum-containedby>
    fn ContainedBy(
        &self,
        Plane0: FXMVECTOR,
        Plane1: FXMVECTOR,
        Plane2: GXMVECTOR,
        Plane3: HXMVECTOR,
        Plane4: HXMVECTOR,
        Plane5: HXMVECTOR,
    ) -> ContainmentType
    {
        // Load origin and orientation of the frustum.
        let mut vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        // Set w of the origin to one so we can dot4 with a plane.
        // TODO: template
        vOrigin = XMVectorInsert(vOrigin, XMVectorSplatOne(), 0, 0, 0, 0, 1);

        // Build the corners of the frustum (in world space).
        let mut RightTop: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
        let mut RightBottom: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
        let mut LeftTop: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
        let mut LeftBottom: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
        let vNear: XMVECTOR = XMVectorReplicatePtr(&self.Near);
        let vFar: XMVECTOR = XMVectorReplicatePtr(&self.Far);

        RightTop = XMVector3Rotate(RightTop, vOrientation);
        RightBottom = XMVector3Rotate(RightBottom, vOrientation);
        LeftTop = XMVector3Rotate(LeftTop, vOrientation);
        LeftBottom = XMVector3Rotate(LeftBottom, vOrientation);

        let Corners0: XMVECTOR = XMVectorMultiplyAdd(RightTop, vNear, vOrigin);
        let Corners1: XMVECTOR = XMVectorMultiplyAdd(RightBottom, vNear, vOrigin);
        let Corners2: XMVECTOR = XMVectorMultiplyAdd(LeftTop, vNear, vOrigin);
        let Corners3: XMVECTOR = XMVectorMultiplyAdd(LeftBottom, vNear, vOrigin);
        let Corners4: XMVECTOR = XMVectorMultiplyAdd(RightTop, vFar, vOrigin);
        let Corners5: XMVECTOR = XMVectorMultiplyAdd(RightBottom, vFar, vOrigin);
        let Corners6: XMVECTOR = XMVectorMultiplyAdd(LeftTop, vFar, vOrigin);
        let Corners7: XMVECTOR = XMVectorMultiplyAdd(LeftBottom, vFar, vOrigin);

        let mut Inside: XMVECTOR = unsafe { uninitialized() };
        let mut Outside: XMVECTOR = unsafe { uninitialized() };

        // Test against each plane.
        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane0, &mut Outside, &mut Inside);

        let mut AnyOutside: XMVECTOR = Outside;
        let mut AllInside: XMVECTOR = Inside;

        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane1, &mut Outside, &mut Inside);

        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane2, &mut Outside, &mut Inside);

        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane3, &mut Outside, &mut Inside);

        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane4, &mut Outside, &mut Inside);

        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectFrustumPlane(Corners0, Corners1, Corners2, Corners3,
            Corners4, Corners5, &Corners6, &Corners7,
            &Plane5, &mut Outside, &mut Inside);

        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        // If the frustum is outside any plane it is outside.
        if (XMVector4EqualInt(AnyOutside, XMVectorTrueInt())) {
            return DISJOINT;
        }

        // If the frustum is inside all planes it is inside.
        if (XMVector4EqualInt(AllInside, XMVectorTrueInt())) {
            return CONTAINS;
        }

        // The frustum is not inside all planes or outside a plane, it may intersect.
        return INTERSECTS;
    }
}

impl CreateFromMatrix for BoundingFrustum {
    fn CreateFromMatrix(Out: &mut Self, Projection: FXMMATRIX) {
        // Corners of the projection frustum in homogenous space.
        const HomogenousPoints: [XMVECTORF32; 6] =
        [
            XMVECTORF32 { f: [  1.0,  0.0, 1.0, 1.0 ] },   // right (at far plane)
            XMVECTORF32 { f: [ -1.0,  0.0, 1.0, 1.0 ] },   // left
            XMVECTORF32 { f: [  0.0,  1.0, 1.0, 1.0 ] },   // top
            XMVECTORF32 { f: [  0.0, -1.0, 1.0, 1.0 ] },   // bottom

            XMVECTORF32 { f: [  0.0,  0.0, 0.0, 1.0 ] },   // near
            XMVECTORF32 { f: [  0.0,  0.0, 1.0, 1.0 ] }    // far
        ];

        let mut Determinant: XMVECTOR = unsafe { uninitialized() };
        let matInverse: XMMATRIX = XMMatrixInverse(Some(&mut Determinant), Projection);

        // Compute the frustum corners in world space.
        let mut Points: [XMVECTOR; 6] = unsafe { uninitialized() };

        //for (size_t i = 0; i < 6; ++i)
        for i in 0 .. 6 
        {
            // Transform point.
            Points[i] = XMVector4Transform(HomogenousPoints[i].v(), matInverse);
        }

        Out.Origin = XMFLOAT3::set(0.0, 0.0, 0.0);
        Out.Orientation = XMFLOAT4::set(0.0, 0.0, 0.0, 1.0);

        // Compute the slopes.
        Points[0] = XMVectorMultiply(Points[0], XMVectorReciprocal(XMVectorSplatZ(Points[0])));
        Points[1] = XMVectorMultiply(Points[1], XMVectorReciprocal(XMVectorSplatZ(Points[1])));
        Points[2] = XMVectorMultiply(Points[2], XMVectorReciprocal(XMVectorSplatZ(Points[2])));
        Points[3] = XMVectorMultiply(Points[3], XMVectorReciprocal(XMVectorSplatZ(Points[3])));

        Out.RightSlope = XMVectorGetX(Points[0]);
        Out.LeftSlope = XMVectorGetX(Points[1]);
        Out.TopSlope = XMVectorGetY(Points[2]);
        Out.BottomSlope = XMVectorGetY(Points[3]);

        // Compute near and far.
        Points[4] = XMVectorMultiply(Points[4], XMVectorReciprocal(XMVectorSplatW(Points[4])));
        Points[5] = XMVectorMultiply(Points[5], XMVectorReciprocal(XMVectorSplatW(Points[5])));

        Out.Near = XMVectorGetZ(Points[4]);
        Out.Far = XMVectorGetZ(Points[5]);
    }
}

//-----------------------------------------------------------------------------
// Build the 6 frustum planes from a frustum.
//
// The intended use for these routines is for fast culling to a view frustum.
// When the volume being tested against a view frustum is small relative to the
// view frustum it is usually either inside all six planes of the frustum
// (CONTAINS) or outside one of the planes of the frustum (DISJOINT). If neither
// of these cases is true then it may or may not be intersecting the frustum
// (INTERSECTS)
//-----------------------------------------------------------------------------
impl BoundingFrustum {
    pub fn GetPlanes(
        &self,
        NearPlane: Option<&mut XMVECTOR>,
        FarPlane: Option<&mut XMVECTOR>,
        RightPlane: Option<&mut XMVECTOR>,
        LeftPlane: Option<&mut XMVECTOR>,
        TopPlane: Option<&mut XMVECTOR>,
        BottomPlane: Option<&mut XMVECTOR>,
    ) {
        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        if let Some(NearPlane) = NearPlane
        {
            let mut vNearPlane: XMVECTOR = XMVectorSet(0.0, 0.0, -1.0, self.Near);
            vNearPlane = internal::XMPlaneTransform(vNearPlane, vOrientation, vOrigin);
            *NearPlane = XMPlaneNormalize(vNearPlane);
        }

        if let Some(FarPlane) = FarPlane
        {
            let mut vFarPlane: XMVECTOR = XMVectorSet(0.0, 0.0, 1.0, -self.Far);
            vFarPlane = internal::XMPlaneTransform(vFarPlane, vOrientation, vOrigin);
            *FarPlane = XMPlaneNormalize(vFarPlane);
        }

        if let Some(RightPlane) = RightPlane
        {
            let mut vRightPlane: XMVECTOR = XMVectorSet(1.0, 0.0, -self.RightSlope, 0.0);
            vRightPlane = internal::XMPlaneTransform(vRightPlane, vOrientation, vOrigin);
            *RightPlane = XMPlaneNormalize(vRightPlane);
        }

        if let Some(LeftPlane) = LeftPlane
        {
            let mut vLeftPlane: XMVECTOR = XMVectorSet(-1.0, 0.0, self.LeftSlope, 0.0);
            vLeftPlane = internal::XMPlaneTransform(vLeftPlane, vOrientation, vOrigin);
            *LeftPlane = XMPlaneNormalize(vLeftPlane);
        }

        if let Some(TopPlane) = TopPlane
        {
            let mut vTopPlane: XMVECTOR = XMVectorSet(0.0, 1.0, -self.TopSlope, 0.0);
            vTopPlane = internal::XMPlaneTransform(vTopPlane, vOrientation, vOrigin);
            *TopPlane = XMPlaneNormalize(vTopPlane);
        }

        if let Some(BottomPlane) = BottomPlane
        {
            let mut vBottomPlane: XMVECTOR = XMVectorSet(0.0, -1.0, self.BottomSlope, 0.0);
            vBottomPlane = internal::XMPlaneTransform(vBottomPlane, vOrientation, vOrigin);
            *BottomPlane = XMPlaneNormalize(vBottomPlane);
        }
    }

    pub fn GetCorners(&self, Corners: &mut [XMFLOAT3; BoundingFrustum::CORNER_COUNT]) {
        // assert(Corners != nullptr);

        // Load origin and orientation of the frustum.
        let vOrigin: XMVECTOR = XMLoadFloat3(&self.Origin);
        let vOrientation: XMVECTOR = XMLoadFloat4(&self.Orientation);

        debug_assert!(internal::XMQuaternionIsUnit(vOrientation));

        // Build the corners of the frustum.
        let vRightTop: XMVECTOR = XMVectorSet(self.RightSlope, self.TopSlope, 1.0, 0.0);
        let vRightBottom: XMVECTOR = XMVectorSet(self.RightSlope, self.BottomSlope, 1.0, 0.0);
        let vLeftTop: XMVECTOR = XMVectorSet(self.LeftSlope, self.TopSlope, 1.0, 0.0);
        let vLeftBottom: XMVECTOR = XMVectorSet(self.LeftSlope, self.BottomSlope, 1.0, 0.0);
        let vNear: XMVECTOR = XMVectorReplicatePtr(&self.Near);
        let vFar: XMVECTOR = XMVectorReplicatePtr(&self.Far);

        // Returns 8 corners position of bounding frustum.
        //     Near    Far
        //    0----1  4----5
        //    |    |  |    |
        //    |    |  |    |
        //    3----2  7----6

        let mut vCorners: [XMVECTOR; Self::CORNER_COUNT] = unsafe { uninitialized() };
        vCorners[0] = XMVectorMultiply(vLeftTop, vNear);
        vCorners[1] = XMVectorMultiply(vRightTop, vNear);
        vCorners[2] = XMVectorMultiply(vRightBottom, vNear);
        vCorners[3] = XMVectorMultiply(vLeftBottom, vNear);
        vCorners[4] = XMVectorMultiply(vLeftTop, vFar);
        vCorners[5] = XMVectorMultiply(vRightTop, vFar);
        vCorners[6] = XMVectorMultiply(vRightBottom, vFar);
        vCorners[7] = XMVectorMultiply(vLeftBottom, vFar);

        for i in 0.. Self::CORNER_COUNT
        {
            let C: XMVECTOR = XMVectorAdd(XMVector3Rotate(vCorners[i], vOrientation), vOrigin);
            XMStoreFloat3(&mut Corners[i], C);
        }
    }
}

pub mod triangle_tests {
    use crate::*;
    use super::*;

    pub fn IntersectsRay(Origin: FXMVECTOR, Direction: FXMVECTOR, V0: FXMVECTOR, V1: GXMVECTOR, V2: HXMVECTOR, Dist: &mut f32) -> bool {
        debug_assert!(internal::XMVector3IsUnit(Direction));

        let Zero: XMVECTOR = XMVectorZero();

        let e1: XMVECTOR = XMVectorSubtract(V1, V0);
        let e2: XMVECTOR = XMVectorSubtract(V2, V0);

        // p = Direction ^ e2;
        let p: XMVECTOR = XMVector3Cross(Direction, e2);

        // det = e1 * p;
        let det: XMVECTOR = XMVector3Dot(e1, p);

        let u: XMVECTOR;
        let v: XMVECTOR;
        let mut t: XMVECTOR;

        if (XMVector3GreaterOrEqual(det, g_RayEpsilon.v()))
        {
            // Determinate is positive (front side of the triangle).
            let s: XMVECTOR = XMVectorSubtract(Origin, V0);

            // u = s * p;
            u = XMVector3Dot(s, p);

            let mut NoIntersection: XMVECTOR = XMVectorLess(u, Zero);
            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(u, det));

            // q = s ^ e1;
            let q: XMVECTOR = XMVector3Cross(s, e1);

            // v = Direction * q;
            v = XMVector3Dot(Direction, q);

            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(v, Zero));
            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(XMVectorAdd(u, v), det));

            // t = e2 * q;
            t = XMVector3Dot(e2, q);

            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(t, Zero));

            if (XMVector4EqualInt(NoIntersection, XMVectorTrueInt()))
            {
                *Dist = 0.0;
                return false;
            }
        }
        else if (XMVector3LessOrEqual(det, g_RayNegEpsilon.v()))
        {
            // Determinate is negative (back side of the triangle).
            let s: XMVECTOR = XMVectorSubtract(Origin, V0);

            // u = s * p;
            u = XMVector3Dot(s, p);

            let mut NoIntersection: XMVECTOR = XMVectorGreater(u, Zero);
            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(u, det));

            // q = s ^ e1;
            let q: XMVECTOR = XMVector3Cross(s, e1);

            // v = Direction * q;
            v = XMVector3Dot(Direction, q);

            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(v, Zero));
            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorLess(XMVectorAdd(u, v), det));

            // t = e2 * q;
            t = XMVector3Dot(e2, q);

            NoIntersection = XMVectorOrInt(NoIntersection, XMVectorGreater(t, Zero));

            if (XMVector4EqualInt(NoIntersection, XMVectorTrueInt()))
            {
                *Dist = 0.0;
                return false;
            }
        }
        else
        {
            // Parallel ray.
            *Dist = 0.0;
            return false;
        }

        t = XMVectorDivide(t, det);

        // (u / det) and (v / dev) are the barycentric cooridinates of the intersection.

        // Store the x-component to *pDist
        XMStoreFloat(Dist, t);

        return true;
    }

    pub fn IntersectsTriangle(
        A0: FXMVECTOR,
        A1: FXMVECTOR,
        A2: FXMVECTOR,
        B0: GXMVECTOR,
        B1: HXMVECTOR,
        B2: HXMVECTOR,
      ) -> bool {
        const SelectY: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_1, XM_SELECT_0, XM_SELECT_0 ] };
        const SelectZ: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_0, XM_SELECT_1, XM_SELECT_0 ] };
        const Select0111: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_0, XM_SELECT_1, XM_SELECT_1, XM_SELECT_1 ] };
        const Select1011: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_1, XM_SELECT_0, XM_SELECT_1, XM_SELECT_1 ] };
        const Select1101: XMVECTORU32 = XMVECTORU32 { u: [ XM_SELECT_1, XM_SELECT_1, XM_SELECT_0, XM_SELECT_1 ] };

        let Zero: XMVECTOR = XMVectorZero();

        // Compute the normal of triangle A.
        let N1: XMVECTOR = XMVector3Cross(XMVectorSubtract(A1, A0), XMVectorSubtract(A2, A0));

        // Assert that the triangle is not degenerate.
        debug_assert!(!XMVector3Equal(N1, Zero));

        // Test points of B against the plane of A.
        let mut BDist: XMVECTOR = XMVector3Dot(N1, XMVectorSubtract(B0, A0));
        BDist = XMVectorSelect(BDist, XMVector3Dot(N1, XMVectorSubtract(B1, A0)), SelectY.v());
        BDist = XMVectorSelect(BDist, XMVector3Dot(N1, XMVectorSubtract(B2, A0)), SelectZ.v());

        // Ensure robustness with co-planar triangles by zeroing small distances.
        let mut BDistIsZeroCR: u32 = unsafe { uninitialized() };
        let BDistIsZero: XMVECTOR = XMVectorGreaterR(&mut BDistIsZeroCR, g_RayEpsilon.v(), XMVectorAbs(BDist));
        BDist = XMVectorSelect(BDist, Zero, BDistIsZero);

        let mut BDistIsLessCR: u32 = unsafe { uninitialized() };
        let BDistIsLess: XMVECTOR = XMVectorGreaterR(&mut BDistIsLessCR, Zero, BDist);

        let mut BDistIsGreaterCR: u32 = unsafe { uninitialized() };
        let BDistIsGreater: XMVECTOR = XMVectorGreaterR(&mut BDistIsGreaterCR, BDist, Zero);

        // If all the points are on the same side we don't intersect.
        if (XMComparisonAllTrue(BDistIsLessCR) || XMComparisonAllTrue(BDistIsGreaterCR)) {
            return false;
        }

        // Compute the normal of triangle B.
        let N2: XMVECTOR = XMVector3Cross(XMVectorSubtract(B1, B0), XMVectorSubtract(B2, B0));

        // Assert that the triangle is not degenerate.
        debug_assert!(!XMVector3Equal(N2, Zero));

        // Test points of A against the plane of B.
        let mut ADist: XMVECTOR = XMVector3Dot(N2, XMVectorSubtract(A0, B0));
        ADist = XMVectorSelect(ADist, XMVector3Dot(N2, XMVectorSubtract(A1, B0)), SelectY.v());
        ADist = XMVectorSelect(ADist, XMVector3Dot(N2, XMVectorSubtract(A2, B0)), SelectZ.v());

        // Ensure robustness with co-planar triangles by zeroing small distances.
        let mut ADistIsZeroCR: u32 = unsafe { uninitialized() };
        let ADistIsZero: XMVECTOR = XMVectorGreaterR(&mut ADistIsZeroCR, g_RayEpsilon.v(), XMVectorAbs(BDist));
        ADist = XMVectorSelect(ADist, Zero, ADistIsZero);

        let mut ADistIsLessCR: u32 = unsafe { uninitialized() };
        let ADistIsLess: XMVECTOR = XMVectorGreaterR(&mut ADistIsLessCR, Zero, ADist);

        let mut ADistIsGreaterCR: u32 = unsafe { uninitialized() };
        let ADistIsGreater: XMVECTOR = XMVectorGreaterR(&mut ADistIsGreaterCR, ADist, Zero);

        // If all the points are on the same side we don't intersect.
        if (XMComparisonAllTrue(ADistIsLessCR) || XMComparisonAllTrue(ADistIsGreaterCR)) {
            return false;
        }

        // Special case for co-planar triangles.
        if (XMComparisonAllTrue(ADistIsZeroCR) || XMComparisonAllTrue(BDistIsZeroCR))
        {
            let mut Axis: XMVECTOR;
            let mut Dist: XMVECTOR;
            let mut MinDist: XMVECTOR;

            // Compute an axis perpindicular to the edge (points out).
            Axis = XMVector3Cross(N1, XMVectorSubtract(A1, A0));
            Dist = XMVector3Dot(Axis, A0);

            // Test points of B against the axis.
            MinDist = XMVector3Dot(B0, Axis);
            MinDist = XMVectorMin(MinDist, XMVector3Dot(B1, Axis));
            MinDist = XMVectorMin(MinDist, XMVector3Dot(B2, Axis));
            if (XMVector4GreaterOrEqual(MinDist, Dist)) {
                return false;
            }

            // Edge (A1, A2)
            Axis = XMVector3Cross(N1, XMVectorSubtract(A2, A1));
            Dist = XMVector3Dot(Axis, A1);

            MinDist = XMVector3Dot(B0, Axis);
            MinDist = XMVectorMin(MinDist, XMVector3Dot(B1, Axis));
            MinDist = XMVectorMin(MinDist, XMVector3Dot(B2, Axis));
            if (XMVector4GreaterOrEqual(MinDist, Dist)) {
                return false;
            }

            // Edge (A2, A0)
            Axis = XMVector3Cross(N1, XMVectorSubtract(A0, A2));
            Dist = XMVector3Dot(Axis, A2);

            MinDist = XMVector3Dot(B0, Axis);
            MinDist = XMVectorMin(MinDist, XMVector3Dot(B1, Axis));
            MinDist = XMVectorMin(MinDist, XMVector3Dot(B2, Axis));
            if (XMVector4GreaterOrEqual(MinDist, Dist)) {
                return false;
            }

            // Edge (B0, B1)
            Axis = XMVector3Cross(N2, XMVectorSubtract(B1, B0));
            Dist = XMVector3Dot(Axis, B0);

            MinDist = XMVector3Dot(A0, Axis);
            MinDist = XMVectorMin(MinDist, XMVector3Dot(A1, Axis));
            MinDist = XMVectorMin(MinDist, XMVector3Dot(A2, Axis));
            if (XMVector4GreaterOrEqual(MinDist, Dist)) {
                return false;
            }

            // Edge (B1, B2)
            Axis = XMVector3Cross(N2, XMVectorSubtract(B2, B1));
            Dist = XMVector3Dot(Axis, B1);

            MinDist = XMVector3Dot(A0, Axis);
            MinDist = XMVectorMin(MinDist, XMVector3Dot(A1, Axis));
            MinDist = XMVectorMin(MinDist, XMVector3Dot(A2, Axis));
            if (XMVector4GreaterOrEqual(MinDist, Dist)) {
                return false;
            }

            // Edge (B2,B0)
            Axis = XMVector3Cross(N2, XMVectorSubtract(B0, B2));
            Dist = XMVector3Dot(Axis, B2);

            MinDist = XMVector3Dot(A0, Axis);
            MinDist = XMVectorMin(MinDist, XMVector3Dot(A1, Axis));
            MinDist = XMVectorMin(MinDist, XMVector3Dot(A2, Axis));
            if (XMVector4GreaterOrEqual(MinDist, Dist)) {
                return false;
            }

            return true;
        }

        //
        // Find the single vertex of A and B (ie the vertex on the opposite side
        // of the plane from the other two) and reorder the edges so we can compute
        // the signed edge/edge distances.
        //
        // if ( (V0 >= 0 && V1 <  0 && V2 <  0) ||
        //      (V0 >  0 && V1 <= 0 && V2 <= 0) ||
        //      (V0 <= 0 && V1 >  0 && V2 >  0) ||
        //      (V0 <  0 && V1 >= 0 && V2 >= 0) ) then V0 is singular;
        //
        // If our singular vertex is not on the positive side of the plane we reverse
        // the triangle winding so that the overlap comparisons will compare the
        // correct edges with the correct signs.
        //
        let ADistIsLessEqual: XMVECTOR = XMVectorOrInt(ADistIsLess, ADistIsZero);
        let ADistIsGreaterEqual: XMVECTOR = XMVectorOrInt(ADistIsGreater, ADistIsZero);

        let AA0: XMVECTOR;
        let AA1: XMVECTOR;
        let AA2: XMVECTOR;
        let bPositiveA: bool;

        if (internal::XMVector3AllTrue(XMVectorSelect(ADistIsGreaterEqual, ADistIsLess, Select0111.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(ADistIsGreater, ADistIsLessEqual, Select0111.v())))
        {
            // A0 is singular, crossing from positive to negative.
            AA0 = A0; AA1 = A1; AA2 = A2;
            bPositiveA = true;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(ADistIsLessEqual, ADistIsGreater, Select0111.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(ADistIsLess, ADistIsGreaterEqual, Select0111.v())))
        {
            // A0 is singular, crossing from negative to positive.
            AA0 = A0; AA1 = A2; AA2 = A1;
            bPositiveA = false;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(ADistIsGreaterEqual, ADistIsLess, Select1011.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(ADistIsGreater, ADistIsLessEqual, Select1011.v())))
        {
            // A1 is singular, crossing from positive to negative.
            AA0 = A1; AA1 = A2; AA2 = A0;
            bPositiveA = true;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(ADistIsLessEqual, ADistIsGreater, Select1011.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(ADistIsLess, ADistIsGreaterEqual, Select1011.v())))
        {
            // A1 is singular, crossing from negative to positive.
            AA0 = A1; AA1 = A0; AA2 = A2;
            bPositiveA = false;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(ADistIsGreaterEqual, ADistIsLess, Select1101.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(ADistIsGreater, ADistIsLessEqual, Select1101.v())))
        {
            // A2 is singular, crossing from positive to negative.
            AA0 = A2; AA1 = A0; AA2 = A1;
            bPositiveA = true;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(ADistIsLessEqual, ADistIsGreater, Select1101.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(ADistIsLess, ADistIsGreaterEqual, Select1101.v())))
        {
            // A2 is singular, crossing from negative to positive.
            AA0 = A2; AA1 = A1; AA2 = A0;
            bPositiveA = false;
        }
        else
        {
            debug_assert!(false);
            return false;
        }

        let BDistIsLessEqual: XMVECTOR = XMVectorOrInt(BDistIsLess, BDistIsZero);
        let BDistIsGreaterEqual: XMVECTOR = XMVectorOrInt(BDistIsGreater, BDistIsZero);

        let BB0: XMVECTOR;
        let BB1: XMVECTOR;
        let BB2: XMVECTOR;
        let bPositiveB: bool;

        if (internal::XMVector3AllTrue(XMVectorSelect(BDistIsGreaterEqual, BDistIsLess, Select0111.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(BDistIsGreater, BDistIsLessEqual, Select0111.v())))
        {
            // B0 is singular, crossing from positive to negative.
            BB0 = B0; BB1 = B1; BB2 = B2;
            bPositiveB = true;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(BDistIsLessEqual, BDistIsGreater, Select0111.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(BDistIsLess, BDistIsGreaterEqual, Select0111.v())))
        {
            // B0 is singular, crossing from negative to positive.
            BB0 = B0; BB1 = B2; BB2 = B1;
            bPositiveB = false;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(BDistIsGreaterEqual, BDistIsLess, Select1011.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(BDistIsGreater, BDistIsLessEqual, Select1011.v())))
        {
            // B1 is singular, crossing from positive to negative.
            BB0 = B1; BB1 = B2; BB2 = B0;
            bPositiveB = true;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(BDistIsLessEqual, BDistIsGreater, Select1011.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(BDistIsLess, BDistIsGreaterEqual, Select1011.v())))
        {
            // B1 is singular, crossing from negative to positive.
            BB0 = B1; BB1 = B0; BB2 = B2;
            bPositiveB = false;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(BDistIsGreaterEqual, BDistIsLess, Select1101.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(BDistIsGreater, BDistIsLessEqual, Select1101.v())))
        {
            // B2 is singular, crossing from positive to negative.
            BB0 = B2; BB1 = B0; BB2 = B1;
            bPositiveB = true;
        }
        else if (internal::XMVector3AllTrue(XMVectorSelect(BDistIsLessEqual, BDistIsGreater, Select1101.v())) ||
            internal::XMVector3AllTrue(XMVectorSelect(BDistIsLess, BDistIsGreaterEqual, Select1101.v())))
        {
            // B2 is singular, crossing from negative to positive.
            BB0 = B2; BB1 = B1; BB2 = B0;
            bPositiveB = false;
        }
        else
        {
            debug_assert!(false);
            return false;
        }

        let Delta0: XMVECTOR;
        let Delta1: XMVECTOR;

        // Reverse the direction of the test depending on whether the singular vertices are
        // the same sign or different signs.
        if (bPositiveA ^ bPositiveB)
        {
            Delta0 = XMVectorSubtract(BB0, AA0);
            Delta1 = XMVectorSubtract(AA0, BB0);
        }
        else
        {
            Delta0 = XMVectorSubtract(AA0, BB0);
            Delta1 = XMVectorSubtract(BB0, AA0);
        }

        // Check if the triangles overlap on the line of intersection between the
        // planes of the two triangles by finding the signed line distances.
        let Dist0: XMVECTOR = XMVector3Dot(Delta0, XMVector3Cross(XMVectorSubtract(BB2, BB0), XMVectorSubtract(AA2, AA0)));
        if (XMVector4Greater(Dist0, Zero)) {
            return false;
        }

        let Dist1: XMVECTOR = XMVector3Dot(Delta1, XMVector3Cross(XMVectorSubtract(BB1, BB0), XMVectorSubtract(AA1, AA0)));
        if (XMVector4Greater(Dist1, Zero)) {
            return false;
        }

        return true;
    }

    pub fn IntersectsPlane(
        V0: FXMVECTOR,
        V1: FXMVECTOR,
        V2: FXMVECTOR,
        Plane: GXMVECTOR,
    ) -> PlaneIntersectionType {
        let One: XMVECTOR = XMVectorSplatOne();

        debug_assert!(internal::XMPlaneIsUnit(Plane));

        // Set w of the points to one so we can dot4 with a plane.
        // TODO: template
        let TV0: XMVECTOR = XMVectorInsert(V0, One, 0, 0, 0, 0, 1);
        let TV1: XMVECTOR = XMVectorInsert(V1, One, 0, 0, 0, 0, 1);
        let TV2: XMVECTOR = XMVectorInsert(V2, One, 0, 0, 0, 0, 1);

        let mut Outside: XMVECTOR = unsafe { uninitialized() };
        let mut Inside: XMVECTOR = unsafe { uninitialized() };

        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, Plane, &mut Outside, &mut Inside);

        // If the triangle is outside any plane it is outside.
        if (XMVector4EqualInt(Outside, XMVectorTrueInt())) {
            return FRONT;
        }

        // If the triangle is inside all planes it is inside.
        if (XMVector4EqualInt(Inside, XMVectorTrueInt())) {
            return BACK;
        }

        // The triangle is not inside all planes or outside a plane it intersects.
        return INTERSECTING;
    }

    /// Test a triangle vs 6 planes (typically forming a frustum).
    pub fn ContainedBy(
        V0: FXMVECTOR,
        V1: FXMVECTOR,
        V2: FXMVECTOR,
        Plane0: GXMVECTOR,
        Plane1: HXMVECTOR,
        Plane2: HXMVECTOR,
        Plane3: CXMVECTOR,
        Plane4: CXMVECTOR,
        Plane5: CXMVECTOR,
    ) -> ContainmentType {
        let One: XMVECTOR = XMVectorSplatOne();

        // Set w of the points to one so we can dot4 with a plane.
        // TODO: template
        let TV0: XMVECTOR = XMVectorInsert(V0, One, 0, 0, 0, 0, 1);
        let TV1: XMVECTOR = XMVectorInsert(V1, One, 0, 0, 0, 0, 1);
        let TV2: XMVECTOR = XMVectorInsert(V2, One, 0, 0, 0, 0, 1);

        let mut Outside: XMVECTOR = unsafe { uninitialized() };
        let mut Inside: XMVECTOR = unsafe { uninitialized() };

        // Test against each plane.
        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, Plane0, &mut Outside, &mut Inside);

        let mut AnyOutside: XMVECTOR = Outside;
        let mut AllInside: XMVECTOR = Inside;

        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, Plane1, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, Plane2, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, *Plane3, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, *Plane4, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        internal::FastIntersectTrianglePlane(TV0, TV1, TV2, *Plane5, &mut Outside, &mut Inside);
        AnyOutside = XMVectorOrInt(AnyOutside, Outside);
        AllInside = XMVectorAndInt(AllInside, Inside);

        // If the triangle is outside any plane it is outside.
        if (XMVector4EqualInt(AnyOutside, XMVectorTrueInt())) {
            return DISJOINT;
        }

        // If the triangle is inside all planes it is inside.
        if (XMVector4EqualInt(AllInside, XMVectorTrueInt())) {
            return CONTAINS;
        }

        // The triangle is not inside all planes or outside a plane, it may intersect.
        return INTERSECTS;
    }
}