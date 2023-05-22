use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub struct Experience {
    pub state: Array1<f32>,
    pub next_state: Array1<f32>,
    pub action: usize,
    pub reward: f32,
    pub done: bool,
}

#[derive(Clone)]
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = thread_rng();
        let samples = self.buffer.as_slices().0;
        let mut indices = (0..self.buffer.len()).collect::<Vec<usize>>();
        indices.shuffle(&mut rng);
        indices.truncate(batch_size);
        indices.into_iter().map(|i| &samples[i]).collect()
    }

    pub fn recent(&self, batch_size: usize) -> Vec<&Experience> {
        let samples = self.buffer.as_slices().0;
        let start = self.buffer.len() - batch_size;
        samples[start..].iter().collect()
    }

    pub fn recent_and_random(&self, batch_size: usize, recent_size: usize) -> Vec<&Experience> {
        let mut rng = thread_rng();
        let samples = self.buffer.as_slices().0;
        let recent_end = self.buffer.len();
        let recent_start = recent_end.saturating_sub(recent_size);
        let mut indices: Vec<usize> = (recent_start..recent_end).collect();
        if indices.len() < batch_size {
            let random_indices: Vec<usize> = (0..recent_start).collect();
            indices.extend(random_indices.choose_multiple(&mut rng, batch_size - indices.len()).cloned());
        }
        indices.shuffle(&mut rng);
        indices.into_iter().map(|i| &samples[i]).collect()
    }
    

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
