//! Random number generation for agents and buffers.
//!
//! Everything that draws randomly takes an `StdRng`. `ThreadRng` wraps an `Rc`, which
//! makes any struct holding one `!Send`, so an agent could not be moved onto a worker
//! thread or into a `Mutex`. `StdRng` is `Send + Sync` and can be seeded, which is what
//! makes a run reproducible.

use rand::SeedableRng;
use rand::rngs::StdRng;

/// A generator seeded from the operating system.
///
/// Used as the serde default for the `rng` field of every agent, since `StdRng` has no
/// `Default` and the generator state is not worth serializing.
pub fn default_rng() -> StdRng {
    StdRng::from_entropy()
}

/// A generator with a fixed seed. Two of these produce the same sequence.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
