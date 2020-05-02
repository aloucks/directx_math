fn main() {
    // https://docs.microsoft.com/en-us/windows/win32/dxmath/ovw-xnamath-reference-directives

    let mut intrinics  = false;

    if cfg!(all(target_feature="sse2", not(feature="no_intrinsics"))) {
        println!("cargo:rustc-cfg=_XM_SSE_INTRINSICS_");
        intrinics = true;

        #[cfg(all(target_feature="sse3", not(feature="no_intrinsics"), not(feature="no_sse3")))]
        println!("cargo:rustc-cfg=_XM_SSE3_INTRINSICS_");

        #[cfg(all(target_feature="sse4.1", not(feature="no_intrinsics"), not(feature="no_sse4")))]
        println!("cargo:rustc-cfg=_XM_SSE4_INTRINSICS_");

        #[cfg(all(target_feature="avx", not(feature="no_intrinsics"), not(feature="no_avx")))]
        println!("cargo:rustc-cfg=_XM_AVX_INTRINSICS_");

        #[cfg(all(target_feature="fma", not(feature="no_intrinsics"), not(feature="no_fma3")))]
        println!("cargo:rustc-cfg=_XM_FMA3_INTRINSICS_");

        #[cfg(all(target_feature="avx2", not(feature="no_intrinsics"), not(feature="no_avx2")))]
        println!("cargo:rustc-cfg=_XM_AVX2_INTRINSICS_");

        #[cfg(feature="favor_intel")]
        println!("cargo:rustc-cfg=_XM_FAVOR_INTEL_");
    } else if cfg!(all(target_arch="arm", not(feature="no_intrinsics"))) {
        // NOTE: ARM intrinics require nightly and don't seem to be fully
        //       implemented/available (or some are missing on docs.rs).
        // println!("cargo:rustc-cfg=_XM_ARM_NEON_INTRINSICS_");
        // intrinics = true;
    }

    if !intrinics {
        println!("cargo:rustc-cfg=_XM_NO_INTRINSICS_");
    }

    if is_nightly() {
        #[cfg(feature="specialization")]
        println!("cargo:rustc-cfg=nightly_specialization");
    }
}

fn is_nightly() -> bool {
    use std::env;
    let rustc = env::var("RUSTC").unwrap();
    let output = std::process::Command::new(rustc).arg("--version").output().unwrap();
    let output = String::from_utf8_lossy(&output.stdout);
    output.contains("nightly")
}