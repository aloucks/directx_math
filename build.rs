fn main() {
    // https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-directives

    let mut intrinics  = false;

    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").expect("CARGO_CFG_TARGET_FEATURE was not defined");
    let target_features = target_features.split(',').collect::<Vec<_>>();
    // eprintln!("available target features: {:?}", target_features);

    if target_features.contains(&"sse2") && !cfg!(feature="no_intrinsics") {
        println!("cargo:rustc-cfg=_XM_SSE_INTRINSICS_");
        intrinics = true;

        if target_features.contains(&"sse3") && !cfg!(feature="no_sse3") {
            println!("cargo:rustc-cfg=_XM_SSE3_INTRINSICS_");
        }

        if target_features.contains(&"sse4.1") && !cfg!(feature="no_sse4") {
            println!("cargo:rustc-cfg=_XM_SSE4_INTRINSICS_");
        }

        if target_features.contains(&"avx") && !cfg!(feature="no_avx") {
            println!("cargo:rustc-cfg=_XM_AVX_INTRINSICS_");
        }

        if target_features.contains(&"f16c") && !cfg!(feature="no_f16c") {
            println!("cargo:rustc-cfg=_XM_F16C_INTRINSICS_");
        }

        if target_features.contains(&"fma") && !cfg!(feature="no_fma") {
            println!("cargo:rustc-cfg=_XM_FMA3_INTRINSICS_");
        }

        if target_features.contains(&"avx2") && !cfg!(feature="no_avx2") {
            println!("cargo:rustc-cfg=_XM_AVX2_INTRINSICS_");
        }

        #[cfg(feature="favor_intel")]
        println!("cargo:rustc-cfg=_XM_FAVOR_INTEL_");
    } else if target_features.contains(&"neon") && !cfg!(feature="no_intrinsics") {
        // NOTE: ARM intrinics require nightly and don't seem to be fully
        //       implemented/available (or some are missing on docs.rs).
        //       https://github.com/aloucks/directx_math/issues/1
        if !cfg!(feature="no_neon") && is_nightly() {
            println!("cargo:rustc-cfg=_XM_ARM_NEON_INTRINSICS_");
            intrinics = true;
        }
    }

    if !intrinics {
        println!("cargo:rustc-cfg=_XM_NO_INTRINSICS_");
    }

    #[cfg(feature="specialization")]
    {
        if is_nightly() {
            println!("cargo:rustc-cfg=nightly_specialization");
        }
    }
}

fn is_nightly() -> bool {
    use std::env;
    let rustc = env::var("RUSTC").unwrap_or_else(|_| String::from("rustc"));
    let output = std::process::Command::new(rustc).arg("--version").output().unwrap();
    let output = String::from_utf8_lossy(&output.stdout);
    output.contains("nightly")
}