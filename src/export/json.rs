use std::fs::File;
use std::io::Write;
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::error::{AthenaError, Result};
use crate::layers::Layer;
use crate::optimizer::OptimizerWrapper;

/// Writes a network's structure and weights to disk.
///
/// This is not ONNX. It is a plain text and JSON dump that external tools can
/// convert; a real ONNX writer would need the protobuf definitions.
pub struct NetworkExporter;

impl NetworkExporter {
    /// Write the architecture and weights as readable text
    pub fn export(network: &NeuralNetwork, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        
        // Write header
        writeln!(file, "# Athena Neural Network Export")?;
        writeln!(file, "# Format: athena network dump")?;
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
            writeln!(file, "activation: {}", activation_op_name(&layer.activation))?;
            
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
                "activation": activation_op_name(&layer.activation),
                "weights": weights,
                "biases": layer.biases.to_vec(),
            });
            
            layers.push(layer_json);
        }
        
        let network_json = json!({
            "format": "athena_network",
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
    
}

/// Name each activation the way ONNX names its operators, so a converter
/// downstream does not have to guess
fn activation_op_name(activation: &Activation) -> &'static str {
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

/// Reads back what `NetworkExporter` wrote.
pub struct NetworkImporter;

impl NetworkImporter {
    /// Read the shape of a network: layer sizes and activations, no weights.
    ///
    /// Use `import_network_json` to get a network that can actually run.
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
        
        // athena_onnx_export was the tag before 0.4.0. It never wrote anything ONNX,
        // so the name changed; files carrying it still read.
        if format != "athena_network" && format != "athena_onnx_export" {
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

    /// Rebuild a runnable network from an exported JSON file.
    ///
    /// `import_json` reads the shape only. This one reads the weight and bias arrays as
    /// well, checks each against the declared sizes and that consecutive layers chain,
    /// and returns a network that produces the same numbers as the one exported.
    ///
    /// The optimizer is not part of the file: pass the one the network should train
    /// with, or a stateless `SGD` if it is only going to run inference.
    pub fn import_network_json(path: &Path, optimizer: OptimizerWrapper) -> Result<NeuralNetwork> {
        use serde_json::Value;

        let file = File::open(path)?;
        let json: Value = serde_json::from_reader(file)?;

        let format = json["format"].as_str().ok_or_else(|| AthenaError::InvalidParameter {
            name: "format".to_string(),
            reason: "Missing format field".to_string(),
        })?;
        if format != "athena_network" && format != "athena_onnx_export" {
            return Err(AthenaError::InvalidParameter {
                name: "format".to_string(),
                reason: format!("Unknown format: {}", format),
            });
        }

        let layers_json = json["model"]["layers"].as_array().ok_or_else(|| {
            AthenaError::InvalidParameter {
                name: "layers".to_string(),
                reason: "Missing layers array".to_string(),
            }
        })?;

        if layers_json.is_empty() {
            return Err(AthenaError::InvalidParameter {
                name: "layers".to_string(),
                reason: "The file declares no layers".to_string(),
            });
        }

        let mut layers = Vec::with_capacity(layers_json.len());
        let mut previous_output: Option<usize> = None;

        for (i, layer_json) in layers_json.iter().enumerate() {
            let field = |name: &str| -> Result<usize> {
                layer_json[name]
                    .as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| AthenaError::InvalidParameter {
                        name: name.to_string(),
                        reason: format!("Missing or invalid {} on layer {}", name, i),
                    })
            };

            let input_size = field("input_size")?;
            let output_size = field("output_size")?;

            if let Some(previous) = previous_output {
                if input_size != previous {
                    return Err(AthenaError::dimension_mismatch(
                        format!("layer {} taking {} inputs", i, previous),
                        format!("{} inputs", input_size),
                    ));
                }
            }

            let activation_str =
                layer_json["activation"]
                    .as_str()
                    .ok_or_else(|| AthenaError::InvalidParameter {
                        name: "activation".to_string(),
                        reason: format!("Missing activation on layer {}", i),
                    })?;
            let activation = onnx_to_activation(activation_str)?;

            let rows = layer_json["weights"].as_array().ok_or_else(|| {
                AthenaError::InvalidParameter {
                    name: "weights".to_string(),
                    reason: format!("Layer {} carries no weights", i),
                }
            })?;

            if rows.len() != input_size {
                return Err(AthenaError::dimension_mismatch(
                    format!("layer {} with {} weight rows", i, input_size),
                    format!("{} rows", rows.len()),
                ));
            }

            let mut weights = Array2::zeros((input_size, output_size));
            for (r, row) in rows.iter().enumerate() {
                let values = row.as_array().ok_or_else(|| AthenaError::InvalidParameter {
                    name: "weights".to_string(),
                    reason: format!("Layer {} row {} is not an array", i, r),
                })?;
                if values.len() != output_size {
                    return Err(AthenaError::dimension_mismatch(
                        format!("layer {} row {} of length {}", i, r, output_size),
                        format!("length {}", values.len()),
                    ));
                }
                for (c, value) in values.iter().enumerate() {
                    weights[[r, c]] =
                        value.as_f64().ok_or_else(|| AthenaError::InvalidParameter {
                            name: "weights".to_string(),
                            reason: format!("Layer {} weight [{}, {}] is not a number", i, r, c),
                        })? as f32;
                }
            }

            let bias_values =
                layer_json["biases"]
                    .as_array()
                    .ok_or_else(|| AthenaError::InvalidParameter {
                        name: "biases".to_string(),
                        reason: format!("Layer {} carries no biases", i),
                    })?;
            if bias_values.len() != output_size {
                return Err(AthenaError::dimension_mismatch(
                    format!("layer {} with {} biases", i, output_size),
                    format!("{} biases", bias_values.len()),
                ));
            }

            let mut biases = Array1::zeros(output_size);
            for (c, value) in bias_values.iter().enumerate() {
                biases[c] = value.as_f64().ok_or_else(|| AthenaError::InvalidParameter {
                    name: "biases".to_string(),
                    reason: format!("Layer {} bias {} is not a number", i, c),
                })? as f32;
            }

            layers.push(
                Layer::new(input_size, output_size, activation)
                    .with_weights(weights)
                    .with_biases(biases),
            );
            previous_output = Some(output_size);
        }

        let network = NeuralNetwork { layers, optimizer };
        network.validate()?;
        Ok(network)
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
        
        NetworkExporter::export_json(&network, &path).unwrap();
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
        
        NetworkExporter::export(&network, &path).unwrap();
        assert!(path.exists());
    }
    
    #[test]
    fn test_activation_conversion() {
        assert_eq!(activation_op_name(&Activation::Relu), "Relu");
        assert_eq!(activation_op_name(&Activation::Linear), "Identity");
        
        let act = onnx_to_activation("Relu").unwrap();
        assert!(matches!(act, Activation::Relu));
    }
}