fn main() {
    #[cfg(feature="no_intrinsics")]
    println!("cargo:rustc-cfg=_XM_NO_INTRINSICS_");

    #[cfg(all(target_feature="avx", not(feature="no_intrinsics"), not(feature="no_avx")))]
    println!("cargo:rustc-cfg=_XM_AVX_INTRINSICS_");

    #[cfg(all(target_feature="avx2", not(feature="no_intrinsics"), not(feature="no_avx2")))]
    println!("cargo:rustc-cfg=_XM_AVX2_INTRINSICS_");

    #[cfg(all(target_feature="fma", not(feature="no_intrinsics"), not(feature="no_fma3")))]
    println!("cargo:rustc-cfg=_XM_FMA3_INTRINSICS_");

    #[cfg(all(target_feature="sse4.1", not(feature="no_intrinsics"), not(feature="no_sse4")))]
    println!("cargo:rustc-cfg=_XM_SSE4_INTRINSICS_");

    #[cfg(all(target_feature="sse3", not(feature="no_intrinsics"), not(feature="no_sse3")))]
    println!("cargo:rustc-cfg=_XM_SSE3_INTRINSICS_");

    #[cfg(all(target_feature="sse2", not(feature="no_intrinsics")))]
    println!("cargo:rustc-cfg=_XM_SSE_INTRINSICS_");

    #[cfg(all(target_arch="arm", not(feature="no_intrinsics")))]
    println!("cargo:rustc-cfg=_XM_NO_INTRINSICS_");
    // NOTE: ARM intrinics require nightly and don't seem to be fully
    //       implemented/available (or they're missing on docs.rs).
    // println!("cargo:rustc-cfg=_XM_ARM_NEON_INTRINSICS_");

    #[cfg(feature="favor_intel")]
    println!("cargo:rustc-cfg=_XM_FAVOR_INTEL_");
}