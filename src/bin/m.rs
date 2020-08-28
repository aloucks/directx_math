use directx_math::*;

fn main() {
    let m = XMMatrixTranslation(2.0, 3.0, 4.0);
    // let mut array: [[f32; 4]; 4] = Default::default();
    let mut array = XMFLOAT4X4::default();
    XMStoreFloat4x4(&mut array, m);
    println!("{:?}", array);
}