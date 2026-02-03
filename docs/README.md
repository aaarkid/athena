# Athena Documentation

## Accessing Documentation

The primary documentation for Athena is now integrated into Rust's documentation system.

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

Once you run `cargo doc --open`, you'll see:

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

### Legacy Markdown Files

The original markdown documentation files in this directory are preserved for reference but have been integrated into the Rust documentation system for better accessibility.