use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};

/// ONNX export functionality for neural networks
pub struct OnnxExporter;

impl OnnxExporter {
    /// Export a neural network to ONNX format
    /// 
    /// Note: This is a simplified implementation that exports the network structure
    /// and weights in a format that can be converted to ONNX using external tools.
    /// A full ONNX implementation would require the onnx protobuf definitions.
    pub fn export(network: &NeuralNetwork, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        
        // Write header
        writeln!(file, "# Athena Neural Network Export")?;
        writeln!(file, "# Format: Simplified ONNX-compatible")?;
        writeln!(file, "# Version: 1.0")?;
        writeln!(file)?;
        
        // Write network architecture
        writeln!(file, "## Network Architecture")?;
        writeln!(file, "num_layers: {}", network.layers.len())?;
        
        for (i, layer) in network.layers.iter().enumerate() {
            writeln!(file)?;
            writeln!(file, "### Layer {}", i)?;
            writeln!(file, "type: Dense")?;
            writeln!(file, "input_size: {}", layer.weights.shape()[0])?;
            writeln!(file, "output_size: {}", layer.weights.shape()[1])?;
            writeln!(file, "activation: {}", activation_to_onnx(&layer.activation))?;
            
            // Write weights
            writeln!(file, "weights_shape: [{}, {}]", layer.weights.shape()[0], layer.weights.shape()[1])?;
            writeln!(file, "weights_data:")?;
            for row in layer.weights.rows() {
                let weights_str: Vec<String> = row.iter().map(|w| format!("{:.6}", w)).collect();
                writeln!(file, "  {}", weights_str.join(", "))?;
            }
            
            // Write biases
            writeln!(file, "biases_shape: [{}]", layer.biases.len())?;
            writeln!(file, "biases_data:")?;
            let biases_str: Vec<String> = layer.biases.iter().map(|b| format!("{:.6}", b)).collect();
            writeln!(file, "  {}", biases_str.join(", "))?;
        }
        
        Ok(())
    }
    
    /// Export network to a JSON format that can be easily converted to ONNX
    pub fn export_json(network: &NeuralNetwork, path: &Path) -> Result<()> {
        use serde_json::json;
        
        let mut layers = Vec::new();
        
        for (i, layer) in network.layers.iter().enumerate() {
            let weights: Vec<Vec<f32>> = layer.weights.rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();
            
            let layer_json = json!({
                "name": format!("layer_{}", i),
                "type": "Dense",
                "input_size": layer.weights.shape()[0],
                "output_size": layer.weights.shape()[1],
                "activation": activation_to_onnx(&layer.activation),
                "weights": weights,
                "biases": layer.biases.to_vec(),
            });
            
            layers.push(layer_json);
        }
        
        let network_json = json!({
            "format": "athena_onnx_export",
            "version": "1.0",
            "model": {
                "name": "athena_network",
                "layers": layers,
            }
        });
        
        let json_str = serde_json::to_string_pretty(&network_json)?;
        let mut file = File::create(path)?;
        file.write_all(json_str.as_bytes())?;
        
        Ok(())
    }
    
    /// Generate ONNX model using tract (when available)
    /// This is a placeholder for future tract integration
    pub fn export_with_tract(_network: &NeuralNetwork, _path: &Path) -> Result<()> {
        Err(AthenaError::InvalidParameter {
            name: "export_with_tract".to_string(),
            reason: "Tract integration not yet implemented. Use export_json instead.".to_string(),
        })
    }
}

/// Convert Athena activation to ONNX operator name
fn activation_to_onnx(activation: &Activation) -> &'static str {
    match activation {
        Activation::Relu => "Relu",
        Activation::Sigmoid => "Sigmoid",
        Activation::Tanh => "Tanh",
        Activation::Linear => "Identity",
        Activation::LeakyRelu { .. } => "LeakyRelu",
        Activation::Elu { .. } => "Elu",
        Activation::Gelu => "Gelu",
    }
}

/// Import functionality for ONNX models
pub struct OnnxImporter;

impl OnnxImporter {
    /// Import a network from JSON format
    pub fn import_json(path: &Path) -> Result<NetworkStructure> {
        use serde_json::Value;
        
        let file = File::open(path)?;
        let json: Value = serde_json::from_reader(file)?;
        
        // Validate format
        let format = json["format"].as_str()
            .ok_or_else(|| AthenaError::InvalidParameter {
                name: "format".to_string(),
                reason: "Missing format field".to_string(),
            })?;
        
        if format != "athena_onnx_export" {
            return Err(AthenaError::InvalidParameter {
                name: "format".to_string(),
                reason: format!("Unknown format: {}", format),
            });
        }
        
        let layers = json["model"]["layers"].as_array()
            .ok_or_else(|| AthenaError::InvalidParameter {
                name: "layers".to_string(),
                reason: "Missing layers array".to_string(),
            })?;
        
        let mut layer_specs = Vec::new();
        
        for layer in layers {
            let input_size = layer["input_size"].as_u64()
                .ok_or_else(|| AthenaError::InvalidParameter {
                    name: "input_size".to_string(),
                    reason: "Missing or invalid input_size".to_string(),
                })? as usize;
            
            let output_size = layer["output_size"].as_u64()
                .ok_or_else(|| AthenaError::InvalidParameter {
                    name: "output_size".to_string(),
                    reason: "Missing or invalid output_size".to_string(),
                })? as usize;
            
            let activation_str = layer["activation"].as_str()
                .ok_or_else(|| AthenaError::InvalidParameter {
                    name: "activation".to_string(),
                    reason: "Missing activation".to_string(),
                })?;
            
            let activation = onnx_to_activation(activation_str)?;
            
            layer_specs.push(LayerSpec {
                input_size,
                output_size,
                activation,
            });
        }
        
        Ok(NetworkStructure { layers: layer_specs })
    }
}

/// Convert ONNX operator name to Athena activation
fn onnx_to_activation(name: &str) -> Result<Activation> {
    match name {
        "Relu" => Ok(Activation::Relu),
        "Sigmoid" => Ok(Activation::Sigmoid),
        "Tanh" => Ok(Activation::Tanh),
        "Identity" => Ok(Activation::Linear),
        "LeakyRelu" => Ok(Activation::LeakyRelu { alpha: 0.01 }),
        "Elu" => Ok(Activation::Elu { alpha: 1.0 }),
        "Gelu" => Ok(Activation::Gelu),
        _ => Err(AthenaError::InvalidParameter {
            name: "activation".to_string(),
            reason: format!("Unknown activation: {}", name),
        }),
    }
}

/// Network structure specification
#[derive(Debug, Clone)]
pub struct NetworkStructure {
    pub layers: Vec<LayerSpec>,
}

/// Layer specification
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Activation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{OptimizerWrapper, SGD};
    use tempfile::tempdir;
    
    #[test]
    fn test_export_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");
        
        let network = NeuralNetwork::new(
            &[2, 3, 1],
            &[Activation::Relu, Activation::Sigmoid],
            OptimizerWrapper::SGD(SGD::new()),
        );
        
        OnnxExporter::export_json(&network, &path).unwrap();
        assert!(path.exists());
    }
    
    #[test]
    fn test_export_text() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.txt");
        
        let network = NeuralNetwork::new(
            &[2, 3, 1],
            &[Activation::Relu, Activation::Sigmoid],
            OptimizerWrapper::SGD(SGD::new()),
        );
        
        OnnxExporter::export(&network, &path).unwrap();
        assert!(path.exists());
    }
    
    #[test]
    fn test_activation_conversion() {
        assert_eq!(activation_to_onnx(&Activation::Relu), "Relu");
        assert_eq!(activation_to_onnx(&Activation::Linear), "Identity");
        
        let act = onnx_to_activation("Relu").unwrap();
        assert!(matches!(act, Activation::Relu));
    }
}