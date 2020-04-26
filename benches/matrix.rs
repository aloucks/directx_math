
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

    fn _XMMatrixDecompose(M: XMMATRIX) -> bool {
        let mut scale = ZERO;
        let mut rotation = ZERO;
        let mut translation = ZERO;
        XMMatrixDecompose(&mut scale, &mut rotation, &mut translation, M)
    }

    // FIXME: The input should be a valid rotation matrix
    bench!(
        bench_XMMatrixDecompose,
        "XMMatrixDecompose",
        _XMMatrixDecompose,
        random_mat4,
    );

    bench!(
        bench_XMMatrixRotationQuaternion,
        "XMMatrixRotationQuaternion",
        XMMatrixRotationQuaternion,
        random_quat,
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

    bench!(
        bench_XMMatrixPerspectiveRH,
        "XMMatrixPerspectiveRH",
        XMMatrixPerspectiveRH,
        random_view_width,
        random_view_height,
        random_near_z,
        random_far_z,
    );

    bench!(
        bench_XMMatrixPerspectiveFovRH,
        "XMMatrixPerspectiveFovRH",
        XMMatrixPerspectiveFovRH,
        random_fov_angle_y,
        random_aspect_ratio,
        random_near_z,
        random_far_z,
    );

    criterion_group!(
        benchmarks,
        bench_XMMatrixIdentity,
        bench_XMMatrixMultiply,
        bench_XMMatrixDecompose,
        bench_XMMatrixRotationQuaternion,
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