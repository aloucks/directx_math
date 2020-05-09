
#![allow(non_snake_case)]

#[macro_use]
mod support;


benchmarks!({
    use criterion::{criterion_group, Criterion};
    use directx_math::collision::*;
    
    use crate::support::*;

    fn _BoundingSphere_Intersects_BoundingSphere(a: BoundingSphere, b: BoundingSphere) -> bool {
        a.IntersectsSphere(&b)
    }

    bench!(
        bench_BoundingSphere_Intersects_BoundingSphere,
        "BoundingSphere::Intersects(BoundingSphere)",
        _BoundingSphere_Intersects_BoundingSphere,
        random_sphere,
        random_sphere,
    );

    fn _BoundingSphere_Intersects_BoundingBox(a: BoundingSphere, b: BoundingBox) -> bool {
        a.IntersectsBox(&b)
    }

    bench!(
        bench_BoundingSphere_Intersects_BoundingBox,
        "BoundingSphere::Intersects(BoundingBox)",
        _BoundingSphere_Intersects_BoundingBox,
        random_sphere,
        random_aabb,
    );

    fn _BoundingSphere_Intersects_BoundingOrientedBox(a: BoundingSphere, b: BoundingOrientedBox) -> bool {
        a.IntersectsOrientedBox(&b)
    }

    bench!(
        bench_BoundingSphere_Intersects_BoundingOrientedBox,
        "BoundingSphere::Intersects(BoundingOrientedBox)",
        _BoundingSphere_Intersects_BoundingOrientedBox,
        random_sphere,
        random_obb,
    );

    fn _BoundingSphere_Intersects_Ray(a: BoundingSphere, ray: Ray) -> bool {
        let (Origin, Direction, mut Dist) = ray;
        a.IntersectsRay((Origin, Direction, &mut Dist))
    }

    bench!(
        bench_BoundingSphere_Intersects_Ray,
        "BoundingSphere::Intersects(Ray)",
        _BoundingSphere_Intersects_Ray,
        random_sphere,
        random_ray,
    );

    // --

    fn _BoundingBox_Intersects_BoundingBox(a: BoundingBox, b: BoundingBox) -> bool {
        a.IntersectsBox(&b)
    }

    bench!(
        bench_BoundingBox_Intersects_BoundingBox,
        "BoundingBox::Intersects(BoundingBox)",
        _BoundingBox_Intersects_BoundingBox,
        random_aabb,
        random_aabb,
    );

    fn _BoundingBox_Intersects_BoundingOrientedBox(a: BoundingBox, b: BoundingOrientedBox) -> bool {
        a.IntersectsOrientedBox(&b)
    }

    bench!(
        bench_BoundingBox_Intersects_BoundingOrientedBox,
        "BoundingBox::Intersects(BoundingOrientedBox)",
        _BoundingBox_Intersects_BoundingOrientedBox,
        random_aabb,
        random_obb,
    );

    fn _BoundingBox_Intersects_Ray(a: BoundingBox, ray: Ray) -> bool {
        let (Origin, Direction, mut Dist) = ray;
        a.IntersectsRay((Origin, Direction, &mut Dist))
    }

    bench!(
        bench_BoundingBox_Intersects_Ray,
        "BoundingBox::Intersects(Ray)",
        _BoundingBox_Intersects_Ray,
        random_aabb,
        random_ray,
    );

    // --

    fn _BoundingOrientedBox_Intersects_BoundingOrientedBox(a: BoundingOrientedBox, b: BoundingOrientedBox) -> bool {
        a.IntersectsOrientedBox(&b)
    }

    bench!(
        bench_BoundingOrientedBox_Intersects_BoundingOrientedBox,
        "BoundingOrientedBox::Intersects(BoundingOrientedBox)",
        _BoundingOrientedBox_Intersects_BoundingOrientedBox,
        random_obb,
        random_obb,
    );

    fn _BoundingOrientedBox_Intersects_Ray(a: BoundingOrientedBox, ray: Ray) -> bool {
        let (Origin, Direction, mut Dist) = ray;
        a.IntersectsRay((Origin, Direction, &mut Dist))
    }

    bench!(
        bench_BoundingOrientedBox_Intersects_Ray,
        "BoundingOrientedBox::Intersects(Ray)",
        _BoundingOrientedBox_Intersects_Ray,
        random_obb,
        random_ray,
    );

    // --

    fn _BoundingSphere_Contains_BoundingSphere(a: BoundingSphere, b: BoundingSphere) -> ContainmentType {
        a.ContainsSphere(&b)
    }

    bench!(
        bench_BoundingSphere_Contains_BoundingSphere,
        "BoundingSphere::Contains(BoundingSphere)",
        _BoundingSphere_Contains_BoundingSphere,
        random_sphere,
        random_sphere,
    );

    fn _BoundingSphere_Contains_BoundingBox(a: BoundingSphere, b: BoundingBox) -> ContainmentType {
        a.ContainsBox(&b)
    }

    bench!(
        bench_BoundingSphere_Contains_BoundingBox,
        "BoundingSphere::Contains(BoundingBox)",
        _BoundingSphere_Contains_BoundingBox,
        random_sphere,
        random_aabb,
    );

    fn _BoundingSphere_Contains_BoundingOrientedBox(a: BoundingSphere, b: BoundingOrientedBox) -> ContainmentType {
        a.ContainsOrientedBox(&b)
    }

    bench!(
        bench_BoundingSphere_Contains_BoundingOrientedBox,
        "BoundingSphere::Contains(BoundingOrientedBox)",
        _BoundingSphere_Contains_BoundingOrientedBox,
        random_sphere,
        random_obb,
    );

    // --

    fn _BoundingBox_Contains_BoundingSphere(a: BoundingBox, b: BoundingSphere) -> ContainmentType {
        a.ContainsSphere(&b)
    }

    bench!(
        bench_BoundingBox_Contains_BoundingSphere,
        "BoundingBox::Contains(BoundingSphere)",
        _BoundingBox_Contains_BoundingSphere,
        random_aabb,
        random_sphere,
    );

    fn _BoundingBox_Contains_BoundingBox(a: BoundingBox, b: BoundingBox) -> ContainmentType {
        a.ContainsBox(&b)
    }

    bench!(
        bench_BoundingBox_Contains_BoundingBox,
        "BoundingBox::Contains(BoundingBox)",
        _BoundingBox_Contains_BoundingBox,
        random_aabb,
        random_aabb,
    );

    fn _BoundingBox_Contains_BoundingOrientedBox(a: BoundingBox, b: BoundingOrientedBox) -> ContainmentType {
        a.ContainsOrientedBox(&b)
    }

    bench!(
        bench_BoundingBox_Contains_BoundingOrientedBox,
        "BoundingBox::Contains(BoundingOrientedBox)",
        _BoundingBox_Contains_BoundingOrientedBox,
        random_aabb,
        random_obb,
    );

    // --

    fn _BoundingOrientedBox_Contains_BoundingSphere(a: BoundingOrientedBox, b: BoundingSphere) -> ContainmentType {
        a.ContainsSphere(&b)
    }

    bench!(
        bench_BoundingOrientedBox_Contains_BoundingSphere,
        "BoundingOrientedBox::Contains(BoundingSphere)",
        _BoundingOrientedBox_Contains_BoundingSphere,
        random_obb,
        random_sphere,
    );

    fn _BoundingOrientedBox_Contains_BoundingBox(a: BoundingOrientedBox, b: BoundingBox) -> ContainmentType {
        a.ContainsBox(&b)
    }

    bench!(
        bench_BoundingOrientedBox_Contains_BoundingBox,
        "BoundingOrientedBox::Contains(BoundingBox)",
        _BoundingOrientedBox_Contains_BoundingBox,
        random_obb,
        random_aabb,
    );

    fn _BoundingOrientedBox_Contains_BoundingOrientedBox(a: BoundingOrientedBox, b: BoundingOrientedBox) -> ContainmentType {
        a.ContainsOrientedBox(&b)
    }

    bench!(
        bench_BoundingOrientedBox_Contains_BoundingOrientedBox,
        "BoundingOrientedBox::Contains(BoundingOrientedBox)",
        _BoundingOrientedBox_Contains_BoundingOrientedBox,
        random_obb,
        random_obb,
    );

    criterion_group!(
        benchmarks,
        bench_BoundingOrientedBox_Contains_BoundingSphere,
        bench_BoundingOrientedBox_Contains_BoundingBox,
        bench_BoundingOrientedBox_Contains_BoundingOrientedBox,
        bench_BoundingSphere_Intersects_BoundingSphere,
        bench_BoundingSphere_Intersects_BoundingBox,
        bench_BoundingSphere_Intersects_BoundingOrientedBox,
        bench_BoundingSphere_Intersects_Ray,
        bench_BoundingBox_Intersects_BoundingBox,
        bench_BoundingBox_Intersects_BoundingOrientedBox,
        bench_BoundingBox_Intersects_Ray,
        bench_BoundingOrientedBox_Intersects_BoundingOrientedBox,
        bench_BoundingOrientedBox_Intersects_Ray,
        bench_BoundingSphere_Contains_BoundingSphere,
        bench_BoundingSphere_Contains_BoundingBox,
        bench_BoundingSphere_Contains_BoundingOrientedBox,
        bench_BoundingBox_Contains_BoundingSphere,
        bench_BoundingBox_Contains_BoundingBox,
        bench_BoundingBox_Contains_BoundingOrientedBox,
    );
});