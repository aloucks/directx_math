
#![allow(non_snake_case)]

#[macro_use]
mod support;


benchmarks!({
    use criterion::{criterion_group, Criterion};
    use directx_math::*;
    
    use crate::support::*;

    bench!(
        bench_XMQuaternionIdentity,
        "XMQuaternionIdentity",
        XMQuaternionIdentity,
    );

    bench!(
        bench_XMQuaternionDot,
        "XMQuaternionDot",
        XMQuaternionDot,
        random_quat,
        random_quat,
    );

    bench!(
        bench_XMQuaternionMultiply,
        "XMQuaternionMultiply",
        XMQuaternionMultiply,
        random_quat,
        random_quat,
    );

    bench!(
        bench_XMQuaternionRotationRollPitchYaw,
        "XMQuaternionRotationRollPitchYaw",
        XMQuaternionRotationRollPitchYaw,
        random_rad,
        random_rad,
        random_rad,
    );

    bench!(
        bench_XMQuaternionSlerp,
        "XMQuaternionSlerp",
        XMQuaternionSlerp,
        random_quat,
        random_quat,
        random_f32,
    );

    bench!(
        bench_XMQuaternionRotationAxis,
        "XMQuaternionRotationAxis",
        XMQuaternionRotationAxis,
        random_vec3,
        random_rad,
    );

    bench!(
        bench_XMQuaternionRotationNormal,
        "XMQuaternionRotationNormal",
        XMQuaternionRotationNormal,
        random_vec3_normalized,
        random_rad,
    );

    criterion_group!(
        benchmarks,
        bench_XMQuaternionIdentity,
        bench_XMQuaternionDot,
        bench_XMQuaternionSlerp,
        bench_XMQuaternionMultiply,
        bench_XMQuaternionRotationRollPitchYaw,
        bench_XMQuaternionRotationNormal,
        bench_XMQuaternionRotationAxis,
    );
});