fn main() {
    // Only configure OpenCL when gpu feature is enabled
    if cfg!(feature = "gpu") {
        // Use local symlink directory
        println!("cargo:rustc-link-search=native=/home/arkid/PROJECTS/athena/target/opencl");
        println!("cargo:rustc-link-lib=OpenCL");
    }
}