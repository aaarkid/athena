use ndarray::{Array1, ArrayView1, concatenate, Axis};
use crate::network::NeuralNetwork;
use crate::activations::Activation;
use crate::optimizer::{OptimizerWrapper, SGD};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Message passing between agents
pub trait CommunicationChannel: Send + Sync {
    type Message;
    
    fn send(&mut self, from: usize, to: usize, message: Self::Message);
    fn receive(&mut self, agent_id: usize) -> Vec<(usize, Self::Message)>;
    fn broadcast(&mut self, from: usize, message: Self::Message);
}

/// Simple broadcast channel for array messages
#[derive(Clone)]
pub struct BroadcastChannel {
    num_agents: usize,
    message_dim: usize,
    messages: Arc<Mutex<HashMap<usize, Vec<(usize, Array1<f32>)>>>>,
}

impl BroadcastChannel {
    pub fn new(num_agents: usize, message_dim: usize) -> Self {
        let mut messages = HashMap::new();
        for i in 0..num_agents {
            messages.insert(i, Vec::new());
        }
        
        Self {
            num_agents,
            message_dim,
            messages: Arc::new(Mutex::new(messages)),
        }
    }
    
    pub fn clear(&mut self) {
        let mut messages = self.messages.lock().unwrap();
        for buffer in messages.values_mut() {
            buffer.clear();
        }
    }
}

impl CommunicationChannel for BroadcastChannel {
    type Message = Array1<f32>;
    
    fn send(&mut self, from: usize, to: usize, message: Self::Message) {
        if to < self.num_agents && message.len() == self.message_dim {
            let mut messages = self.messages.lock().unwrap();
            if let Some(buffer) = messages.get_mut(&to) {
                buffer.push((from, message));
            }
        }
    }
    
    fn receive(&mut self, agent_id: usize) -> Vec<(usize, Self::Message)> {
        let mut messages = self.messages.lock().unwrap();
        messages.get_mut(&agent_id)
            .map(|buffer| {
                let received = buffer.clone();
                buffer.clear();
                received
            })
            .unwrap_or_default()
    }
    
    fn broadcast(&mut self, from: usize, message: Self::Message) {
        if message.len() == self.message_dim {
            let mut messages = self.messages.lock().unwrap();
            for (to, buffer) in messages.iter_mut() {
                if *to != from {
                    buffer.push((from, message.clone()));
                }
            }
        }
    }
}

/// Agent with communication capabilities
pub struct CommunicatingAgent<C: CommunicationChannel> {
    id: usize,
    message_encoder: NeuralNetwork,
    message_decoder: NeuralNetwork,
    comm_channel: C,
    state_dim: usize,
    message_dim: usize,
}

impl<C: CommunicationChannel<Message = Array1<f32>>> CommunicatingAgent<C> {
    pub fn new(
        id: usize,
        state_dim: usize,
        message_dim: usize,
        comm_channel: C,
    ) -> Self {
        let encoder_activations = vec![Activation::Relu, Activation::Linear];
        let decoder_activations = vec![Activation::Relu, Activation::Linear];
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        
        // Encoder: state -> message
        let message_encoder = NeuralNetwork::new(
            &[state_dim, 64, message_dim],
            &encoder_activations,
            optimizer.clone(),
        );
        
        // Decoder: messages -> state augmentation
        let message_decoder = NeuralNetwork::new(
            &[message_dim, 64, state_dim],
            &decoder_activations,
            optimizer,
        );
        
        Self {
            id,
            message_encoder,
            message_decoder,
            comm_channel,
            state_dim,
            message_dim,
        }
    }
    
    /// Process incoming messages
    pub fn encode_messages(&mut self, messages: &[(usize, Array1<f32>)]) -> Array1<f32> {
        if messages.is_empty() {
            return Array1::zeros(self.state_dim);
        }
        
        // Average all messages (simple aggregation)
        let mut sum = Array1::zeros(self.message_dim);
        for (_, msg) in messages {
            sum = sum + msg;
        }
        let avg_message = sum / messages.len() as f32;
        
        // Decode to state augmentation
        self.message_decoder.forward(avg_message.view())
    }
    
    /// Generate message from current state
    pub fn generate_message(&mut self, state: ArrayView1<f32>) -> Array1<f32> {
        self.message_encoder.forward(state)
    }
    
    /// Act with communication
    pub fn act_with_communication<F>(
        &mut self,
        state: ArrayView1<f32>,
        act_fn: F,
    ) -> (usize, Option<Array1<f32>>)
    where
        F: FnOnce(ArrayView1<f32>) -> usize,
    {
        // Receive messages
        let messages = self.comm_channel.receive(self.id);
        
        // Encode messages
        let encoded_messages = self.encode_messages(&messages);
        
        // Combine with state
        let augmented_state = concatenate![Axis(0), state, encoded_messages.view()];
        
        // Get action
        let action = act_fn(augmented_state.view());
        
        // Generate message
        let message = self.generate_message(state);
        
        (action, Some(message))
    }
    
    /// Send message to specific agent
    pub fn send_message(&mut self, to: usize, message: Array1<f32>) {
        self.comm_channel.send(self.id, to, message);
    }
    
    /// Broadcast message to all agents
    pub fn broadcast_message(&mut self, message: Array1<f32>) {
        self.comm_channel.broadcast(self.id, message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_broadcast_channel() {
        let mut channel = BroadcastChannel::new(3, 4);
        
        // Agent 0 broadcasts
        let msg = array![1.0, 2.0, 3.0, 4.0];
        channel.broadcast(0, msg.clone());
        
        // Agents 1 and 2 should receive
        let msgs1 = channel.receive(1);
        let msgs2 = channel.receive(2);
        
        assert_eq!(msgs1.len(), 1);
        assert_eq!(msgs2.len(), 1);
        assert_eq!(msgs1[0].0, 0); // From agent 0
        assert_eq!(msgs1[0].1, msg);
        
        // Agent 0 should not receive its own broadcast
        let msgs0 = channel.receive(0);
        assert_eq!(msgs0.len(), 0);
    }
    
    #[test]
    fn test_point_to_point_communication() {
        let mut channel = BroadcastChannel::new(3, 4);
        
        // Agent 0 sends to agent 1
        let msg = array![1.0, 2.0, 3.0, 4.0];
        channel.send(0, 1, msg.clone());
        
        // Only agent 1 should receive
        let msgs1 = channel.receive(1);
        assert_eq!(msgs1.len(), 1);
        assert_eq!(msgs1[0].1, msg);
        
        // Others should not receive
        let msgs0 = channel.receive(0);
        let msgs2 = channel.receive(2);
        assert_eq!(msgs0.len(), 0);
        assert_eq!(msgs2.len(), 0);
    }
    
    #[test]
    fn test_communicating_agent() {
        let channel = BroadcastChannel::new(2, 4);
        let mut agent = CommunicatingAgent::new(0, 8, 4, channel);
        
        let state = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Act with no messages
        let (action, msg) = agent.act_with_communication(
            state.view(),
            |_s| 1, // Dummy action function
        );
        
        assert_eq!(action, 1);
        assert!(msg.is_some());
        assert_eq!(msg.unwrap().len(), 4);
    }
}