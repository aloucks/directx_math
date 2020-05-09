
#![allow(dead_code)]
#![allow(non_snake_case)]

#[macro_export]
macro_rules! benchmarks {
    ($benchmarks:tt) => {
        #[cfg(feature="benchmarks")]
        mod benchmark $benchmarks

        #[cfg(feature="benchmarks")]
        fn main() {
            criterion::criterion_main!(benchmark::benchmarks);

            pub fn criterion_main() {
                main();
            }

            criterion_main()
        }
        
        #[cfg(not(feature="benchmarks"))]
        fn main() {
            println!("Run with: cargo bench --features benchmarks")
        }
    };
}

#[macro_export]
macro_rules! bench {
    (@input => $rng:expr, $arg:expr) => {
        $arg($rng)
    };
    (@arg => $value:ident, $arg:expr) => {
        $value
    };
    (@arg => $value:ident, ref $arg:expr) => {
        &$value
    };
    (@op => $op:ident, $inputs:expr $(,)*) => {{
        $op()
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr) => {{
        let arg1 = *$inputs;
        $op(arg1)
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr) => {{
        let (arg1, arg2) = *$inputs;
        $op(arg1, bench!(@arg => arg2, $arg2))
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr, $arg3:expr) => {{
        let (arg1, arg2, arg3) = *$inputs;
        $op(arg1, arg2, arg3)
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr) => {{
        let (arg1, arg2, arg3, arg4) = *$inputs;
        $op(arg1, arg2, arg3, arg4)
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr) => {{
        let (arg1, arg2, arg3, arg4, arg5) = *$inputs;
        $op(arg1, arg2, arg3, arg4, arg5)
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr, $arg6:expr) => {{
        let (arg1, arg2, arg3, arg4, arg5, arg6) = *$inputs;
        $op(arg1, arg2, arg3, arg4, arg5, arg6)
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr, $arg6:expr, $arg7:expr) => {{
        let (arg1, arg2, arg3, arg4, arg5, arg6, arg7) = *$inputs;
        $op(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    }};
    (@op => $op:ident, $inputs:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr, $arg6:expr, $arg7:expr, $arg8:expr) => {{
        let (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) = *$inputs;
        $op(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    }};
    ($name:ident, $desc:expr, $op:ident, $($args:expr),* $(,)*) => {
        pub fn $name(c: &mut Criterion) {
            use pcg::Pcg;

            const SIZE: usize = 8192;
            let rng = &mut Pcg::new(0, 0);
            let inputs = (0..SIZE).map(|_: usize| ($(bench!(@input => rng, $args)),*)).collect::<Vec<_>>();
            //let mut outputs = (0..SIZE).map(|_: usize| $op($(bench!(@input => rng, $args)),*)).collect::<Vec<_>>();
            let mut i = 0;
            c.bench_function($desc, |b| {
                b.iter(|| {
                    i = (i + 1) & (SIZE - 1);
                    unsafe {
                        let _inputs = inputs.get_unchecked(i);
                        //*outputs.get_unchecked_mut(i) = bench!(@op => $op, inputs, $($args),*);
                        let output = bench!(@op => $op, _inputs, $($args),*);
                        criterion::black_box(output);
                        //criterion::black_box(inputs);
                    }
                })
            });
            criterion::black_box(inputs);
            //criterion::black_box(outputs);
            criterion::black_box(rng);
        }
    };
}

#[cfg(feature="benchmarks")]
pub use inner::*;

#[cfg(feature="benchmarks")]
mod inner {
    use std::f32::consts::PI;
    use pcg::Pcg;
    use rand_core::RngCore;
    use directx_math::*;
    use directx_math::collision::*;

    pub fn random_f32(rng: &mut Pcg) -> f32 {
        (rng.next_u32() & 0xffffff) as f32 / 16777216.0
    }

    pub fn random_rad(rng: &mut Pcg) -> f32 {
        -PI + random_f32(rng) * 2.0 * PI
    }

    pub fn random_vec2(rng: &mut Pcg) -> XMVECTOR {
        XMVectorSet(
            random_f32(rng),
            random_f32(rng),
            0.0,
            0.0,
        )
    }

    pub fn random_vec3(rng: &mut Pcg) -> XMVECTOR {
        XMVectorSet(
            random_f32(rng),
            random_f32(rng),
            random_f32(rng),
            0.0,
        )
    }

    pub fn random_vec3_normalized(rng: &mut Pcg) -> XMVECTOR {
        XMVector3Normalize(random_vec3(rng))
    }

    pub fn random_vec4(rng: &mut Pcg) -> XMVECTOR {
        XMVectorSet(
            random_f32(rng),
            random_f32(rng),
            random_f32(rng),
            random_f32(rng),
        )
    }

    pub fn random_quat(rng: &mut Pcg) -> XMVECTOR {
        XMQuaternionRotationRollPitchYaw(
            random_rad(rng),
            random_rad(rng),
            random_rad(rng),
        )
    }

    pub fn random_mat4(rng: &mut Pcg) -> XMMATRIX {
        XMMatrixSet(
            random_f32(rng), random_f32(rng), random_f32(rng), random_f32(rng),
            random_f32(rng), random_f32(rng), random_f32(rng), random_f32(rng),
            random_f32(rng), random_f32(rng), random_f32(rng), random_f32(rng),
            random_f32(rng), random_f32(rng), random_f32(rng), random_f32(rng),
        )
    }

    pub fn quat_identity(_rng: &mut Pcg) -> XMVECTOR {
        XMQuaternionIdentity()
    }

    pub fn vec_zero(_rng: &mut Pcg) -> XMVECTOR {
        XMVectorSet(0.0, 0.0, 0.0, 0.0)
    }

    pub fn vec_one(_rng: &mut Pcg) -> XMVECTOR {
        XMVectorSet(1.0, 1.0, 1.0, 1.0)
    }

    fn range_f32(rng: &mut Pcg, low: f32, high: f32) -> f32 {
        let r = random_f32(rng);
        (r * (high - low)) + low
    }

    fn range_u32(rng: &mut Pcg, low: u32, high: u32) -> u32 {
        let r = random_f32(rng);
        let (low, high) = (low as f32, high as f32);
        let v = (r * (high - low)) + low;
        let v: u32 = v.round() as u32;
        assert!(v >= low as u32);
        assert!(v <= high as u32);
        v
    }

    pub fn random_vec4_permute(rng: &mut Pcg) -> u32 {
        range_u32(rng, 0, 7)
    }


    pub fn random_view_width(rng: &mut Pcg) -> f32 {
        range_f32(rng, 1024.0, 2560.0)
    }

    pub fn random_view_height(rng: &mut Pcg) -> f32 {
        range_f32(rng, 768.0, 1600.0)
    }

    pub fn random_near_z(rng: &mut Pcg) -> f32 {
        range_f32(rng, 0.001, 1.0)
    }

    pub fn random_far_z(rng: &mut Pcg) -> f32 {
        let a = range_f32(rng, 100.0, 10_000.0);
        assert!(a >= 100.0);
        assert!(a <= 10_000.0);
        a
    }

    pub fn random_fov_angle_y(rng: &mut Pcg) -> f32 {
        range_f32(rng, 45.0f32.to_radians(), 90.0f32.to_radians())
    }

    pub fn random_aspect_ratio(rng: &mut Pcg) -> f32 {
        random_view_width(rng) / random_view_height(rng)
    }

    pub fn random_obb(rng: &mut Pcg) -> BoundingOrientedBox {
        let extent = range_f32(rng, 0.5, 2.0);
        let center = range_f32(rng, -10.0, 10.0);
        let Center = XMFLOAT3::set(center, center, center);
        let Extents = XMFLOAT3::set(extent, extent, extent);
        let mut Orientation = XMFLOAT4::set(0.0, 0.0, 0.0, 0.0);
        XMStoreFloat4(&mut Orientation, random_quat(rng));
        BoundingOrientedBox {
            Center,
            Extents,
            Orientation,
        }
    }

    pub fn random_aabb(rng: &mut Pcg) -> BoundingBox {
        let extent = range_f32(rng, 0.5, 2.0);
        let center = range_f32(rng, -10.0, 10.0);
        let Center = XMFLOAT3::set(center, center, center);
        let Extents = XMFLOAT3::set(extent, extent, extent);
        BoundingBox {
            Center,
            Extents,
        }
    }

    pub fn random_sphere(rng: &mut Pcg) -> BoundingSphere {
        let Radius = range_f32(rng, 0.5, 2.0);
        let center = range_f32(rng, -10.0, 10.0);
        let Center = XMFLOAT3::set(center, center, center);
        BoundingSphere {
            Center,
            Radius,
        }
    }

    pub fn random_ray(rng: &mut Pcg) -> Ray {
        let origin = range_f32(rng, -10.0, 10.0);
        let Origin = XMVectorSet(origin, origin, origin, 0.0);
        let Direction = XMVector3Normalize(random_vec3(rng));
        (Origin, Direction)
    }

    pub const ZERO: XMVECTOR = unsafe { XMVECTORF32 { f: [0.0, 0.0, 0.0, 0.0] }.v };
}

