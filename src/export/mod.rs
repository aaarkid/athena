//! Writing a trained network out to disk in a form other tools can read.

pub mod json;

pub use json::{NetworkExporter, NetworkImporter, NetworkStructure, LayerSpec};
