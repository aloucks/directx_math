use regex::Regex;
use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let r = Regex::new("(v[^\\s|^\\(]+_[f|u|i|s]\\d+)\\(")?;
    let filenames = [
        "../DirectXMath/Inc/DirectXMath.h",
        "../DirectXMath/Inc/DirectXMathVector.inl",
        "../DirectXMath/Inc/DirectXMathMatrix.inl",
        "../DirectXMath/Inc/DirectXMathMisc.inl",
        "../DirectXMath/Inc/DirectXMathConvert.inl",
    ];
    let mut fn_names = BTreeMap::new();
    for filename in filenames.iter() {
        let text = fs::read_to_string(filename)?;
        for captures in r.captures_iter(&text) {
            if let Some(fn_name) = captures.get(1) {
                let count = fn_names.entry(fn_name.as_str().to_owned()).or_insert(0);
                *count += 1;
            }
        }
    }

    let _ = Command::new("curl")
        .args(&[
            "-LRJo",
            "target/index.html",
            "https://doc.rust-lang.org/core/arch/aarch64/index.html",
        ])
        .output()?;

    let index = fs::read_to_string("target/index.html")?;
    let mut status = Vec::new();
    let mut missing = 0;
    for (fn_name, _) in fn_names.iter() {
        let exists = index.contains(fn_name);
        status.push((fn_name, exists));
        if !exists {
            missing += 1;
        }
    }
    println!(
        "### Neon Intrinsics ({}/{})",
        fn_names.len() - missing,
        fn_names.len()
    );
    for (fn_name, exists) in status {
        println!(
            "- [{}] `{}` ({})",
            if exists { "x" } else { " " },
            fn_name,
            fn_names[fn_name]
        );
    }
    Ok(())
}