use std::env;
use std::path::PathBuf;
use std::process::Command;

const LOGQ: &str = "48";

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let labrador_dir =
        env::var("BIN_NTT_LABRADOR_DIR").unwrap_or_else(|_| format!("{manifest_dir}/labrador"));
    println!("cargo:rerun-if-env-changed=BIN_NTT_LABRADOR_DIR");

    if !PathBuf::from(format!("{labrador_dir}/Makefile")).exists() {
        panic!("the labrador submodule is missing at {labrador_dir}; run `git submodule update --init`");
    }

    // The LaBRADOR objects hard-code LOGQ (modulus, K, and the labrador<LOGQ>_ symbol
    // prefix), so a LOGQ change has to invalidate every object file, not just relink.
    let libobj_dir = format!("{labrador_dir}/libobj");
    let stamp_path = format!("{libobj_dir}/.bin_ntt_logq_stamp");
    let stamp_matches = std::fs::read_to_string(&stamp_path)
        .map(|s| s.trim() == LOGQ)
        .unwrap_or(false);
    if !stamp_matches {
        let _ = Command::new("make")
            .args(["-C", &labrador_dir, "clean"])
            .status();
    }

    let status = Command::new("make")
        .env("CC", std::env::var("CC").unwrap_or_else(|_| "cc".into()))
        .args([
            "-C",
            &labrador_dir,
            &format!("LOGQ={LOGQ}"),
            "liblabrador.a",
        ])
        .status()
        .expect("failed to invoke make for liblabrador.a");
    assert!(
        status.success(),
        "make -C {labrador_dir} LOGQ={LOGQ} liblabrador.a failed"
    );
    std::fs::create_dir_all(&libobj_dir)
        .expect("failed to create labrador libobj dir for the LOGQ stamp");
    std::fs::write(&stamp_path, LOGQ).expect("failed to write the LOGQ stamp");
    println!("cargo:rerun-if-changed={labrador_dir}");

    let csrc_dir = format!("{manifest_dir}/csrc");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&csrc_dir)
        .expect("failed to read csrc/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "c"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no C sources in {csrc_dir}");

    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".into());
    let mut objects = Vec::new();
    for src in &sources {
        let obj = out_dir.join(src.file_stem().unwrap()).with_extension("o");
        let status = Command::new(&cc)
            .args([
                "-std=c2x",
                "-O3",
                "-march=native",
                "-fPIE",
                "-mtune=native",
                "-fwrapv",
                "-Wall",
                "-Wextra",
                "-Wno-unused-function",
                &format!("-DLOGQ={LOGQ}"),
                "-I",
                &labrador_dir,
                "-I",
                &csrc_dir,
                "-c",
            ])
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to invoke gcc");
        assert!(status.success(), "gcc failed to compile {}", src.display());
        objects.push(obj);
    }
    for entry in std::fs::read_dir(&csrc_dir).expect("failed to read csrc/") {
        println!("cargo:rerun-if-changed={}", entry.unwrap().path().display());
    }

    let c_lib = out_dir.join("libbin_ntt_c.a");
    let _ = std::fs::remove_file(&c_lib);
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&c_lib)
        .args(&objects)
        .status()
        .expect("failed to invoke ar");
    assert!(status.success(), "ar failed to archive libbin_ntt_c.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={labrador_dir}");
    println!("cargo:rustc-link-lib=static=bin_ntt_c");
    println!("cargo:rustc-link-lib=static=labrador");
    println!("cargo:rustc-link-lib=m");
}
