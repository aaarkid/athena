#![allow(dead_code)]
#![allow(unused_macros)]

#[macro_use]
pub mod agent;
pub mod network;

pub mod replay_buffer;
pub mod optimizer;

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use ndarray::arr2;
    use ndarray::array;
    use crate::agent::DqnAgent;
    use crate::network::{Activation, Layer, NeuralNetwork};
    use crate::optimizer::Adam;
    use crate::optimizer::Optimizer;
    use crate::optimizer::{OptimizerWrapper, SGD};
    use crate::replay_buffer::Experience;
    use crate::replay_buffer::ReplayBuffer;

    #[test]
    fn test_layer_creation() {
        let input_size = 3;
        let output_size = 2;
        let activation = Activation::Relu;
        let layer = Layer::new(input_size, output_size, activation);

        assert_eq!(layer.weights.shape(), [input_size, output_size]);
        assert_eq!(layer.biases.shape(), [output_size]);
    }

    #[test]
    fn test_neural_network_creation() {
        let layer_sizes = &[3, 4, 2];
        let activations = &[Activation::Relu, Activation::Relu];
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let network = NeuralNetwork::new(layer_sizes, activations, optimizer);

        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].weights.shape(), [3, 4]);
        assert_eq!(network.layers[0].biases.shape(), [4]);
        assert_eq!(network.layers[1].weights.shape(), [4, 2]);
        assert_eq!(network.layers[1].biases.shape(), [2]);
    }

    #[test]
    fn test_forward_pass() {
        let layer_sizes = &[3, 4, 2];
        let activations = &[Activation::Relu, Activation::Relu];
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);

        let input = arr1(&[1.0, 2.0, 3.0]);
        let output = network.forward(input.view());

        assert_eq!(output.shape(), [2]);
    }

    #[test]
    fn test_train_minibatch() {
        let layer_sizes = &[2, 4, 1];
        let activations = &[Activation::Relu, Activation::Relu];
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut network = NeuralNetwork::new(layer_sizes, activations, optimizer);

        let inputs = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
        ]);

        let targets = arr2(&[
            [1.0],
            [2.0],
        ]);

        network.train_minibatch(inputs.view(), targets.view(), 0.01);

        let new_output1 = network.forward(inputs.row(0).view());
        let new_output2 = network.forward(inputs.row(1).view());

        assert_eq!(new_output1.shape(), [1]);
        assert_eq!(new_output2.shape(), [1]);

    }

    #[test]
    fn test_sgd_update_weights() {
        let mut sgd = SGD::new();
        let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
        let gradients = array![[0.1, 0.2], [0.3, 0.4]];
        let learning_rate = 0.01;
    
        sgd.update_weights(&mut weights, &gradients, learning_rate);
    
        let expected_weights = array![[0.999, 0.998], [0.997, 0.996]];
        assert_eq!(weights, expected_weights);
    }

    #[test]
    fn test_sgd_update_biases() {
        let mut sgd = SGD::new();
        let mut biases = array![1.0, 1.0];
        let gradients = array![0.1, 0.2];
        let learning_rate = 0.01;
    
        sgd.update_biases(&mut biases, &gradients, learning_rate);
    
        let expected_biases = array![0.999, 0.998];
        assert_eq!(biases, expected_biases);
    }

    #[test]
    fn test_adam_new() {
        let layers = vec![Layer::new(2, 2, Activation::Relu)];
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
    
        let adam = Adam::new(&layers, beta1, beta2, epsilon);
    
        assert_eq!(adam.beta1, beta1);
        assert_eq!(adam.beta2, beta2);
        assert_eq!(adam.epsilon, epsilon);
        assert_eq!(adam.t, 1);
    }

    #[test]
    fn test_adam_update_weights() {
        let layers = vec![Layer::new(2, 2, Activation::Relu)];
        let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);

        let mut weights = array![[1.0, 1.0], [1.0, 1.0]];
        let gradients = array![[0.1, 0.2], [0.3, 0.4]];
        let learning_rate = 0.01;

        adam.update_weights(&mut weights, &gradients, learning_rate);

        let expected_weights = array![
            [0.99, 0.99],
            [0.99, 0.99]
        ];
        assert_eq!(weights, expected_weights);
    }

    #[test]
    fn test_adam_update_biases() {
        let layers = vec![Layer::new(2, 2, Activation::Relu)];
        let mut adam = Adam::new(&layers, 0.9, 0.999, 1e-8);

        let mut biases = array![1.0, 1.0];
        let gradients = array![0.1, 0.2];
        let learning_rate = 0.01;

        adam.update_biases(&mut biases, &gradients, learning_rate);

        let expected_biases = array![0.99, 0.99];
        assert_eq!(biases, expected_biases);
    }

    fn create_test_agent() -> DqnAgent {
        let layer_sizes = [4, 32, 2];
        let epsilon = 0.5;
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        DqnAgent::new(&layer_sizes, epsilon, optimizer)
    }

    #[test]
    fn test_dqn_agent_new() {
        let agent = create_test_agent();
        assert_eq!(agent.epsilon, 0.5);
        assert_eq!(agent.network.layers.len(), 2);
    }

    #[test]
    fn test_dqn_agent_act() {
        let mut agent = create_test_agent();
        let state = array![0.0, 0.5, 1.0, 0.5];
        let action = agent.act(state.view());
        assert!(action < agent.network.layers.last().unwrap().biases.len());
    }

    #[test]
    fn test_dqn_agent_update_epsilon() {
        let mut agent = create_test_agent();
        agent.update_epsilon(0.1);
        assert_eq!(agent.epsilon, 0.1);
    }

    #[test]
    fn test_dqn_agent_train_on_batch1() {
        let mut agent = create_test_agent();
        let experience = Experience {
            state: array![0.0, 0.5, 1.0, 0.5],
            action: 0,
            reward: 1.0,
            next_state: array![0.1, 0.6, 0.9, 0.4],
            done: false,
        };
        let gamma = 0.99;
        let learning_rate = 0.001;
        agent.train_on_batch(&[&experience], gamma, learning_rate);
    }

    #[test]
    fn test_dqn_agent_train_on_batch2() {
        let layer_sizes = [2, 32, 2];
        let epsilon = 0.5;
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let mut agent = DqnAgent::new(&layer_sizes, epsilon, optimizer);
        let experience = Experience {
            state: array![0.5, -0.5],
            action: 0,
            reward: 1.0,
            next_state: array![0.6, -0.4],
            done: false,
        };
        let gamma = 0.99;
        let learning_rate = 0.01;
        agent.train_on_batch(&[&experience], gamma, learning_rate);
        // Verify that weights and biases have been updated (which would be implementation-specific)
    }

    #[test]
    fn test_replay_buffer_add_and_sample() {
        let mut replay_buffer = ReplayBuffer::new(10);
        let experience = Experience {
            state: array![0.5, -0.5],
            action: 0,
            reward: 1.0,
            next_state: array![0.6, -0.4],
            done: false,
        };
        replay_buffer.add(experience.clone());
        assert_eq!(replay_buffer.len(), 1);
        let sample = replay_buffer.sample(1);
        assert_eq!(sample[0], &experience);
    }
}
