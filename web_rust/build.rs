fn main() {
    // Tell cargo to rerun this build script if the Python code changes
    println!("cargo:rerun-if-changed=../src/slimgest");
}
