
#![allow(non_snake_case)]

#[macro_use]
mod support;


benchmarks!({
    use criterion::{criterion_group, Criterion};
    use directx_math::*;
    
    use crate::support::*;

    bench!(
        bench_XMMatrixIdentity,
        "XMMatrixIdentity",
        XMMatrixIdentity,
    );

    fn _XMMatrixMultiply(M1: XMMATRIX, M2: XMMATRIX) -> XMMATRIX {
        XMMatrixMultiply(M1, &M2)
    }

    bench!(
        bench_XMMatrixMultiply,
        "XMMatrixMultiply",
        _XMMatrixMultiply,
        random_mat4,
        random_mat4,
    );

    bench!(
        bench_XMMatrixTranspose,
        "XMMatrixTranspose",
        XMMatrixTranspose,
        random_mat4,
    );

    fn _XMMatrixMultiplyTranspose(M1: XMMATRIX, M2: XMMATRIX) -> XMMATRIX {
        XMMatrixMultiplyTranspose(M1, &M2)
    }

    bench!(
        bench_XMMatrixMultiplyTranspose,
        "XMMatrixMultiplyTranspose",
        _XMMatrixMultiplyTranspose,
        random_mat4,
        random_mat4,
    );

    fn _XMMatrixInverse(M: XMMATRIX) -> XMMATRIX {
        let mut determinant = XMVectorZero();
        XMMatrixInverse(&mut determinant, M)
    }

    bench!(
        bench_XMMatrixInverse,
        "XMMatrixInverse",
        _XMMatrixInverse,
        random_mat4,
    );

    bench!(
        bench_XMMatrixDeterminant,
        "XMMatrixDeterminant",
        XMMatrixDeterminant,
        random_mat4,
    );

    bench!(
        bench_XMMatrixTransformation,
        "XMMatrixTransformation",
        XMMatrixTransformation,
        vec_zero,
        quat_identity,
        vec_one,
        vec_zero,
        random_quat,
        random_vec3,
    );

    bench!(
        bench_XMMatrixLookAtRH,
        "XMMatrixLookAtRH",
        XMMatrixLookAtRH,
        random_vec3,
        random_vec3,
        random_vec3,
    );

    // FIXME: The inputs should use values clamped to valid ranges
    bench!(
        bench_XMMatrixPerspectiveRH,
        "XMMatrixPerspectiveRH",
        XMMatrixPerspectiveRH,
        random_f32,
        random_f32,
        random_f32,
        random_f32,
    );

    // FIXME: The inputs should use values clamped to valid ranges
    bench!(
        bench_XMMatrixPerspectiveFovRH,
        "XMMatrixPerspectiveFovRH",
        XMMatrixPerspectiveFovRH,
        random_f32,
        random_f32,
        random_f32,
        random_f32,
    );

    criterion_group!(
        benchmarks,
        bench_XMMatrixIdentity,
        bench_XMMatrixMultiply,
        bench_XMMatrixTranspose,
        bench_XMMatrixMultiplyTranspose,
        bench_XMMatrixInverse,
        bench_XMMatrixDeterminant,
        bench_XMMatrixTransformation,
        bench_XMMatrixLookAtRH,
        bench_XMMatrixPerspectiveRH,
        bench_XMMatrixPerspectiveFovRH,
    );
});