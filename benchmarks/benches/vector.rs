
#![allow(non_snake_case)]

#[macro_use]
mod support;


benchmarks!({
    use criterion::{criterion_group, Criterion};
    use directx_math::*;
    
    use crate::support::*;

    bench!(
        bench_XMVectorSet,
        "XMVectorSet",
        XMVectorSet,
        random_f32,
        random_f32,
        random_f32,
        random_f32,
    );

    bench!(
        bench_XMVectorSplatX,
        "XMVectorSplatX",
        XMVectorSplatX,
        random_vec4,
    );

    bench!(
        bench_XMVectorSplatY,
        "XMVectorSplatY",
        XMVectorSplatY,
        random_vec4,
    );

    bench!(
        bench_XMVectorSplatZ,
        "XMVectorSplatZ",
        XMVectorSplatZ,
        random_vec4,
    );

    bench!(
        bench_XMVectorSplatW,
        "XMVectorSplatW",
        XMVectorSplatW,
        random_vec4,
    );

    bench!(
        bench_XMVectorRound,
        "XMVectorRound",
        XMVectorRound,
        random_vec4,
    );

    bench!(
        bench_XMVectorMultiplyAdd,
        "XMVectorMultiplyAdd",
        XMVectorMultiplyAdd,
        random_vec4,
        random_vec4,
        random_vec4,
    );

    bench!(
        bench_XMVectorSin,
        "XMVectorSin",
        XMVectorSin,
        random_vec4,
    );

    bench!(
        bench_XMVectorSinEst,
        "XMVectorSinEst",
        XMVectorSinEst,
        random_vec4,
    );

    bench!(
        bench_XMVectorCos,
        "XMVectorCos",
        XMVectorCos,
        random_vec4,
    );

    bench!(
        bench_XMVectorCosEst,
        "XMVectorCosEst",
        XMVectorCosEst,
        random_vec4,
    );

    bench!(
        bench_XMVectorTan,
        "XMVectorTan",
        XMVectorTan,
        random_vec4,
    );

    bench!(
        bench_XMVectorTanEst,
        "XMVectorTanEst",
        XMVectorTanEst,
        random_vec4,
    );

    bench!(
        bench_XMVectorASin,
        "XMVectorASin",
        XMVectorASin,
        random_vec4,
    );

    bench!(
        bench_XMVectorASinEst,
        "XMVectorASinEst",
        XMVectorASinEst,
        random_vec4,
    );

    bench!(
        bench_XMVectorACos,
        "XMVectorACos",
        XMVectorACos,
        random_vec4,
    );

    bench!(
        bench_XMVectorACosEst,
        "XMVectorACosEst",
        XMVectorACosEst,
        random_vec4,
    );

    bench!(
        bench_XMVectorATan,
        "XMVectorATan",
        XMVectorATan,
        random_vec4,
    );

    bench!(
        bench_XMVectorATanEst,
        "XMVectorATanEst",
        XMVectorATanEst,
        random_vec4,
    );

    bench!(
        bench_XMVectorATan2,
        "XMVectorATan2",
        XMVectorATan2,
        random_vec4,
        random_vec4,
    );

    bench!(
        bench_XMVectorATan2Est,
        "XMVectorATan2Est",
        XMVectorATan2Est,
        random_vec4,
        random_vec4,
    );

    bench!(
        bench_XMVectorLerp,
        "XMVectorLerp",
        XMVectorLerp,
        random_vec4,
        random_vec4,
        random_f32,
    );

    bench!(
        bench_XMVector3Dot,
        "XMVector3Dot",
        XMVector3Dot,
        random_vec3,
        random_vec3,
    );

    bench!(
        bench_XMVector3Cross,
        "XMVector3Cross",
        XMVector3Dot,
        random_vec3,
        random_vec3,
    );

    bench!(
        bench_XMVector3Length,
        "XMVector3Length",
        XMVector3Length,
        random_vec3,
    );

    bench!(
        bench_XMVector3LengthEst,
        "XMVector3LengthEst",
        XMVector3LengthEst,
        random_vec3,
    );

    bench!(
        bench_XMVector3Normalize,
        "XMVector3Normalize",
        XMVector3Normalize,
        random_vec3,
    );

    bench!(
        bench_XMVector3NormalizeEst,
        "XMVector3NormalizeEst",
        XMVector3NormalizeEst,
        random_vec3,
    );

    bench!(
        bench_XMVector3ClampLength,
        "XMVector3ClampLength",
        XMVector3ClampLength,
        random_vec3,
        random_near_z,
        random_far_z,
    );

    bench!(
        bench_XMVector3Orthogonal,
        "XMVector3Orthogonal",
        XMVector3Orthogonal,
        random_vec3,
    );

    bench!(
        bench_XMVector3AngleBetweenNormals,
        "XMVector3AngleBetweenNormals",
        XMVector3AngleBetweenNormals,
        random_vec3_normalized,
        random_vec3_normalized,
    );

    bench!(
        bench_XMVector3AngleBetweenVectors,
        "XMVector3AngleBetweenVectors",
        XMVector3AngleBetweenVectors,
        random_vec3,
        random_vec3,
    );

    bench!(
        bench_XMVector3Rotate,
        "XMVector3Rotate",
        XMVector3Rotate,
        random_vec3,
        random_quat,
    );

    bench!(
        bench_XMVector3Transform,
        "XMVector3Transform",
        XMVector3Transform,
        random_vec3,
        random_mat4,
    );

    criterion_group!(
        benchmarks,
        bench_XMVectorSet,
        bench_XMVectorSplatX,
        bench_XMVectorSplatY,
        bench_XMVectorSplatZ,
        bench_XMVectorSplatW,
        bench_XMVectorRound,
        bench_XMVectorMultiplyAdd,
        bench_XMVectorSin,
        bench_XMVectorSinEst,
        bench_XMVectorCos,
        bench_XMVectorCosEst,
        bench_XMVectorTan,
        bench_XMVectorTanEst,
        bench_XMVectorASin,
        bench_XMVectorASinEst,
        bench_XMVectorACos,
        bench_XMVectorACosEst,
        bench_XMVectorATan,
        bench_XMVectorATanEst,
        bench_XMVectorATan2,
        bench_XMVectorATan2Est,
        bench_XMVectorLerp,
        bench_XMVector3Dot,
        bench_XMVector3Cross,
        bench_XMVector3Length,
        bench_XMVector3LengthEst,
        bench_XMVector3Normalize,
        bench_XMVector3NormalizeEst,
        bench_XMVector3ClampLength,
        bench_XMVector3Orthogonal,
        bench_XMVector3AngleBetweenNormals,
        bench_XMVector3AngleBetweenVectors,
        bench_XMVector3Rotate,
        bench_XMVector3Transform,
    );
});