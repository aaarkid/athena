# Athena Documentation Guide

## Overview

Athena now provides comprehensive documentation through Rust's built-in documentation system. All tutorials, guides, and API references are available in one unified location.

## Accessing Documentation

### Local Documentation
```bash
# Build and open documentation in your browser
cargo doc --open

# Build with all features enabled
cargo doc --all-features --open
```

### Online Documentation
Once published to crates.io, documentation will be available at:
- https://docs.rs/athena

## Documentation Structure

### 1. Main Landing Page (`src/lib.rs`)
The crate root documentation provides:
- 📚 Documentation Hub with organized links
- 🚀 Getting Started section
- 🎯 Core Concepts overview
- 🔧 Advanced Topics
- 📖 Reference Guides

### 2. Tutorials Module (`src/tutorials.rs`)
Comprehensive tutorials are now integrated into the Rust docs:
- **Getting Started** - Complete beginner's guide with working examples
- **Advanced Features** - Custom layers, parallel training, GPU acceleration
- **Best Practices** - Code organization and performance tips
- **Performance Guide** - Optimization techniques and benchmarking
- **Algorithm Guide** - Comparison and selection of RL algorithms

### 3. Module Documentation
Each module has comprehensive documentation:
- Module overview with key concepts
- Working examples in doc comments
- API reference for all public items
- Cross-references to related modules

## Key Features

### Integrated Tutorials
All markdown tutorials from `docs/` have been converted to Rust documentation:
- Syntax-highlighted code examples
- Runnable example code (with `cargo test --doc`)
- Better integration with API docs
- Type-checked examples

### Navigation
The documentation provides multiple ways to navigate:
1. **Documentation Hub** in the main page
2. **Module hierarchy** in the sidebar
3. **Search functionality** built into rustdoc
4. **Cross-references** between related topics

### Examples in Documentation
Every major component includes examples:
```rust
//! ```rust
//! use athena::agent::DqnAgent;
//! // Example code here...
//! ```
```

## Benefits of This Approach

1. **Single Source of Truth** - All documentation in one place
2. **Always Up-to-Date** - Documentation is part of the code
3. **Type-Checked Examples** - Examples are compiled and tested
4. **Better Discoverability** - IDE integration and search
5. **Version-Specific** - Documentation matches exact version

## For Contributors

When adding new features:
1. Add module-level documentation with `//!`
2. Document all public items with `///`
3. Include examples in doc comments
4. Update relevant tutorial sections
5. Cross-reference related modules

## Viewing Different Sections

After running `cargo doc --open`:
1. Click on "athena" in the sidebar
2. Navigate to "Modules" section
3. Explore:
   - `tutorials` - All guides and tutorials
   - `agent` - RL agents documentation
   - `algorithms` - Advanced algorithms
   - `network` - Neural network docs
   - And more...

## Legacy Documentation

The original markdown files are preserved in `docs/` for reference:
- `docs/algorithms_guide.md`
- `docs/best_practices.md`
- `docs/performance_guide.md`
- `docs/tutorial_advanced.md`
- `docs/tutorial_getting_started.md`

These have been integrated into the Rust documentation system for better accessibility.