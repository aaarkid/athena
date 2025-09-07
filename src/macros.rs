/// A macro to create a new `Layer`.
///
/// # Examples
///
/// ```
/// use athena::layers::Layer;
/// use athena::activations::Activation;
/// use athena::create_layer;
/// let layer = create_layer!(4, 32, Activation::Relu);
/// ```
///
/// This will create a new `Layer` with an input size of 4, an output size of 32, and uses the ReLU activation function.
#[macro_export]
macro_rules! create_layer {
    ($input_size:expr, $output_size:expr, $activation:expr) => {
        $crate::layers::Layer::new($input_size, $output_size, $activation)
    };
}

/// A macro to create a new `NeuralNetwork`.
///
/// # Examples
///
/// ```
/// use athena::optimizer::{OptimizerWrapper, SGD};
/// use athena::create_network;
/// use athena::activations::Activation;
/// use athena::network::NeuralNetwork;
/// use athena::layers::Layer;
/// let optimizer = OptimizerWrapper::SGD(SGD::new());
/// let network = create_network!(optimizer,
///     (4, 32, Activation::Relu), 
///     (32, 2, Activation::Linear)
/// );
/// ```
///
/// This will create a new `NeuralNetwork` with two layers: the first layer has an input size of 4, an 
/// output size of 32, and uses the ReLU activation function; the second layer has an input size of 32 
/// (matching the output size of the first layer), an output size of 2, and uses the Linear activation function.
#[macro_export]
macro_rules! create_network {
    ($optimizer:expr, $( ($input_size:expr, $output_size:expr, $activation:expr) ),* ) => {
        {
            let layers = vec![$( $crate::layers::Layer::new($input_size, $output_size, $activation) ),*];
            $crate::network::NeuralNetwork { layers, optimizer: $optimizer }
        }
    }
}