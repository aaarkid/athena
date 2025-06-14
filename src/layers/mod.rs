pub mod traits;
pub mod dense;

pub use traits::Layer as LayerTrait;
pub use dense::{DenseLayer, Layer};