fn main() {
    // Only configure OpenCL when gpu feature is enabled
    if cfg!(feature = "gpu") {
        // Platform-specific OpenCL configuration
        if cfg!(target_os = "windows") {
            // Common OpenCL SDK locations on Windows
            let possible_paths = vec![
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\lib\\x64",
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\lib\\x64",
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\lib\\x64",
                "C:\\Program Files (x86)\\Intel\\OpenCL SDK\\lib\\x64",
                "C:\\Program Files\\Intel\\OpenCL SDK\\lib\\x64",
                "C:\\Windows\\System32",
                "C:\\Program Files (x86)\\OCL_SDK_Light\\lib\\x86_64",
            ];
            
            for path in &possible_paths {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
            
            // Try to link OpenCL
            println!("cargo:rustc-link-lib=OpenCL");
        } else if cfg!(target_os = "linux") {
            // Linux paths
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native=/usr/local/lib");
            println!("cargo:rustc-link-lib=OpenCL");
        } else if cfg!(target_os = "macos") {
            // macOS has OpenCL as a framework
            println!("cargo:rustc-link-lib=framework=OpenCL");
        }
    }
}