
#![allow(dead_code)]

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
}

