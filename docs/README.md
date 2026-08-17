# Athena Documentation

## Accessing Documentation

The API documentation lives in Rust's own documentation system.

### View Documentation

```bash
# Build and open documentation
cargo doc --open

# Build with all features
cargo doc --all-features --open

# Build without dependencies (faster)
cargo doc --no-deps --open
```

### Documentation Structure

`cargo doc --open` gives you:

1. **Main Page** - Overview and documentation hub with links
2. **Modules Section** - Click to expand and see all modules
3. **Tutorials Module** - Contains all guides:
   - `getting_started` - Complete beginner's guide
   - `advanced` - Advanced features tutorial
   - `best_practices` - Coding guidelines
   - `performance` - Optimization guide
   - `algorithms` - RL algorithm comparison

### Navigation Tips

1. Use the search box to find specific items
2. Click on module names to see sub-modules
3. Click on struct/trait names for detailed documentation
4. Look for "Examples" sections in each item

### Markdown guides

The guides in this directory are included in the crate documentation under
`athena::docs`, with `#[doc = include_str!]`. That means **every fenced `rust` block in
them is compiled by `cargo test`**, so a sample cannot drift from the API without the
build failing.

If you add a block, tag output and math fences `text`: rustdoc treats a bare fence as
Rust and tries to compile it.

`athena::tutorials` holds a second, older copy of some of the same material as module
documentation.