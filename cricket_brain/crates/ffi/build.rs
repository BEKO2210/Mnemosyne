// SPDX-License-Identifier: AGPL-3.0-only
use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_header = crate_dir.join("include").join("cricket_brain.h");

    if let Some(parent) = out_header.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .unwrap_or_else(|_| cbindgen::Config::default());

    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("unable to generate C header via cbindgen");
    bindings.write_to_file(&out_header);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
