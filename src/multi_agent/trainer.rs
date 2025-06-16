use crate::agent::DqnAgent;
use crate::multi_agent::environment::MultiAgentEnvironment;
use crate::replay_buffer::{ReplayBuffer, Experience};
#[cfg(feature = "action-masking")]
use crate::agent::MaskedAgent;
use rand::prelude::*;
use std::collections::HashMap;

/// Self-play trainer for multi-agent systems
pub struct SelfPlayTrainer {
    agent_pool: Vec<DqnAgent>,
    update_interval: usize,
    sampling_strategy: SamplingStrategy,
    elo_ratings: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    /// Sample uniformly from agent pool
    Uniform,
    /// Sample based on skill similarity
    Prioritized { temperature: f32 },
    /// League play style sampling
    League { main_prob: f32, main_exploit_prob: f32 },
}

#[derive(Default, Debug)]
pub struct TrainingMetrics {
    pub episodes_completed: usize,
    pub average_reward: f32,
    pub win_rates: HashMap<usize, f32>,
    pub elo_changes: Vec<f32>,
}

impl SelfPlayTrainer {
    pub fn new(
        initial_agent: DqnAgent,
        pool_size: usize,
        update_interval: usize,
        sampling_strategy: SamplingStrategy,
    ) -> Self {
        let mut agent_pool = Vec::with_capacity(pool_size);
        let mut elo_ratings = Vec::with_capacity(pool_size);
        
        // Initialize pool with copies of initial agent
        // Create pool by deserializing/serializing the initial agent
        let serialized = bincode::serialize(&initial_agent).unwrap();
        for _ in 0..pool_size {
            let agent: DqnAgent = bincode::deserialize(&serialized).unwrap();
            agent_pool.push(agent);
            elo_ratings.push(1500.0); // Standard ELO starting rating
        }
        
        Self {
            agent_pool,
            update_interval,
            sampling_strategy,
            elo_ratings,
        }
    }
    
    /// Train agents through self-play
    pub fn train<E>(
        &mut self,
        env: &mut E,
        episodes: usize,
        batch_size: usize,
    ) -> TrainingMetrics
    where
        E: MultiAgentEnvironment,
        E::Observation: Into<ndarray::Array1<f32>> + Clone,
        E::Action: From<usize>,
    {
        let mut metrics = TrainingMetrics::default();
        let mut replay_buffers: Vec<ReplayBuffer> = (0..env.num_agents())
            .map(|_| ReplayBuffer::new(100000))
            .collect();
        
        for episode in 0..episodes {
            // Sample agents for this episode
            let agent_indices = self.sample_agents(env.num_agents());
            
            // Play episode
            let episode_rewards = self.play_episode(
                env,
                &agent_indices,
                &mut replay_buffers,
            );
            
            // Update metrics
            metrics.episodes_completed += 1;
            let total_reward: f32 = episode_rewards.values().sum();
            metrics.average_reward = (metrics.average_reward * episode as f32 + total_reward) 
                / (episode + 1) as f32;
            
            // Update agents
            if episode % self.update_interval == 0 && episode > 0 {
                self.update_agents(&agent_indices, &mut replay_buffers, batch_size);
                
                // Update ELO ratings based on game outcome
                if episode_rewards.len() == 2 {
                    self.update_elo_ratings(&agent_indices, &episode_rewards);
                }
            }
        }
        
        metrics
    }
    
    fn play_episode<E>(
        &mut self,
        env: &mut E,
        agent_indices: &[usize],
        replay_buffers: &mut [ReplayBuffer],
    ) -> HashMap<usize, f32>
    where
        E: MultiAgentEnvironment,
        E::Observation: Into<ndarray::Array1<f32>> + Clone,
        E::Action: From<usize>,
    {
        let mut observations = env.reset();
        let mut episode_experiences: Vec<Vec<Experience>> = vec![Vec::new(); env.num_agents()];
        let mut episode_rewards = HashMap::new();
        
        while !env.is_terminal() {
            let active = env.active_agents();
            let mut actions = Vec::new();
            
            for &agent_id in &active {
                let obs_array: ndarray::Array1<f32> = env.get_observation(agent_id).into();
                let agent_idx = agent_indices[agent_id];
                let agent = &mut self.agent_pool[agent_idx];
                
                #[cfg(feature = "action-masking")]
                let action = {
                    let mask = env.legal_actions(agent_id);
                    agent.act_masked(obs_array.view(), &mask)
                };
                
                #[cfg(not(feature = "action-masking"))]
                let action = agent.act(obs_array.view()).unwrap_or(0);
                
                actions.push((agent_id, E::Action::from(action)));
            }
            
            let transition = env.step(&actions);
            
            // Store experiences
            for &agent_id in &active {
                let obs_array: ndarray::Array1<f32> = observations[agent_id].clone().into();
                let next_obs_array: ndarray::Array1<f32> = 
                    transition.observations.get(&agent_id)
                        .cloned()
                        .unwrap_or_else(|| observations[agent_id].clone())
                        .into();
                
                let _action = actions.iter()
                    .find(|(id, _)| *id == agent_id)
                    .map(|(_, act)| act)
                    .unwrap();
                
                episode_experiences[agent_id].push(Experience {
                    state: obs_array,
                    action: 0, // TODO: properly convert action
                    reward: transition.rewards.get(&agent_id).copied().unwrap_or(0.0),
                    next_state: next_obs_array,
                    done: transition.done,
                });
                
                *episode_rewards.entry(agent_id).or_insert(0.0) += 
                    transition.rewards.get(&agent_id).copied().unwrap_or(0.0);
            }
            
            observations = (0..env.num_agents())
                .map(|i| transition.observations.get(&i).cloned()
                    .unwrap_or_else(|| observations[i].clone()))
                .collect();
        }
        
        // Add experiences to replay buffers
        for (agent_id, experiences) in episode_experiences.into_iter().enumerate() {
            for exp in experiences {
                replay_buffers[agent_id].add(exp);
            }
        }
        
        episode_rewards
    }
    
    fn sample_agents(&self, num_needed: usize) -> Vec<usize> {
        let mut rng = thread_rng();
        
        match &self.sampling_strategy {
            SamplingStrategy::Uniform => {
                (0..num_needed)
                    .map(|_| rng.gen_range(0..self.agent_pool.len()))
                    .collect()
            }
            SamplingStrategy::Prioritized { temperature } => {
                // Sample based on ELO similarity
                let mut agents = vec![rng.gen_range(0..self.agent_pool.len())];
                
                for _ in 1..num_needed {
                    let reference_elo = self.elo_ratings[agents[0]];
                    let weights: Vec<f32> = self.elo_ratings.iter()
                        .map(|&elo| {
                            let diff = (elo - reference_elo).abs();
                            (-diff / temperature).exp()
                        })
                        .collect();
                    
                    let sum: f32 = weights.iter().sum();
                    let mut cumsum = 0.0;
                    let r = rng.gen::<f32>() * sum;
                    
                    for (i, &w) in weights.iter().enumerate() {
                        cumsum += w;
                        if cumsum > r {
                            agents.push(i);
                            break;
                        }
                    }
                }
                
                agents
            }
            SamplingStrategy::League { main_prob, main_exploit_prob } => {
                // Simplified league play sampling
                (0..num_needed)
                    .map(|_| {
                        let r = rng.gen::<f32>();
                        if r < *main_prob {
                            0 // Main agent
                        } else if r < main_prob + main_exploit_prob {
                            1 // Main exploiter
                        } else {
                            rng.gen_range(2..self.agent_pool.len())
                        }
                    })
                    .collect()
            }
        }
    }
    
    fn update_agents(
        &mut self,
        agent_indices: &[usize],
        replay_buffers: &mut [ReplayBuffer],
        batch_size: usize,
    ) {
        for (buffer_id, &agent_idx) in agent_indices.iter().enumerate() {
            if replay_buffers[buffer_id].len() >= batch_size {
                let batch = replay_buffers[buffer_id].sample(batch_size);
                let agent = &mut self.agent_pool[agent_idx];
                
                // Convert to references for train_on_batch
                let exp_refs: Vec<&Experience> = batch.iter().map(|e| &**e).collect();
                let _ = agent.train_on_batch(&exp_refs, 0.99, 0.001);
                
                // Update target network periodically
                agent.update_target_network();
            }
        }
    }
    
    fn update_elo_ratings(
        &mut self,
        agent_indices: &[usize],
        rewards: &HashMap<usize, f32>,
    ) {
        if agent_indices.len() != 2 {
            return; // ELO only works for 2-player games
        }
        
        let k = 32.0; // ELO K-factor
        let idx0 = agent_indices[0];
        let idx1 = agent_indices[1];
        
        let r0 = self.elo_ratings[idx0];
        let r1 = self.elo_ratings[idx1];
        
        // Expected scores
        let e0 = 1.0 / (1.0 + 10_f32.powf((r1 - r0) / 400.0));
        let e1 = 1.0 / (1.0 + 10_f32.powf((r0 - r1) / 400.0));
        
        // Actual scores (based on rewards)
        let reward0 = rewards.get(&0).copied().unwrap_or(0.0);
        let reward1 = rewards.get(&1).copied().unwrap_or(0.0);
        
        let s0 = if reward0 > reward1 { 1.0 } else if reward0 < reward1 { 0.0 } else { 0.5 };
        let s1 = 1.0 - s0;
        
        // Update ratings
        self.elo_ratings[idx0] += k * (s0 - e0);
        self.elo_ratings[idx1] += k * (s1 - e1);
    }
    
    /// Get agent by index
    pub fn get_agent(&self, index: usize) -> Option<&DqnAgent> {
        self.agent_pool.get(index)
    }
    
    /// Get current ELO ratings
    pub fn get_elo_ratings(&self) -> &[f32] {
        &self.elo_ratings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{SGD, OptimizerWrapper};
    
    #[test]
    fn test_self_play_trainer_creation() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = DqnAgent::new(&[4, 32, 32, 2], 0.1, optimizer, 100, false);
        
        let trainer = SelfPlayTrainer::new(
            agent,
            4,
            10,
            SamplingStrategy::Uniform,
        );
        
        assert_eq!(trainer.agent_pool.len(), 4);
        assert_eq!(trainer.elo_ratings.len(), 4);
        assert!(trainer.elo_ratings.iter().all(|&r| r == 1500.0));
    }
    
    #[test]
    fn test_agent_sampling() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = DqnAgent::new(&[4, 32, 32, 2], 0.1, optimizer, 100, false);
        
        let trainer = SelfPlayTrainer::new(
            agent,
            10,
            10,
            SamplingStrategy::Uniform,
        );
        
        let agents = trainer.sample_agents(4);
        assert_eq!(agents.len(), 4);
        assert!(agents.iter().all(|&idx| idx < 10));
    }
    
    #[test]
    fn test_elo_update() {
        let optimizer = OptimizerWrapper::SGD(SGD::new());
        let agent = DqnAgent::new(&[4, 32, 32, 2], 0.1, optimizer, 100, false);
        
        let mut trainer = SelfPlayTrainer::new(
            agent,
            2,
            10,
            SamplingStrategy::Uniform,
        );
        
        let mut rewards = HashMap::new();
        rewards.insert(0, 10.0);
        rewards.insert(1, 5.0);
        
        let initial_elo = trainer.elo_ratings.clone();
        trainer.update_elo_ratings(&[0, 1], &rewards);
        
        // Winner's rating should increase
        assert!(trainer.elo_ratings[0] > initial_elo[0]);
        // Loser's rating should decrease
        assert!(trainer.elo_ratings[1] < initial_elo[1]);
    }
}