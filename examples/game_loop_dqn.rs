//! The whole path a game needs: act every frame, learn from what happened, save, reload.
//!
//! Run with `cargo run --release --example game_loop_dqn`.
//!
//! The environment is a 7 by 7 grid. The agent starts at a corner and the goal sits at
//! the opposite one. Reaching it pays 1.0 and ends the episode; every step costs 0.01,
//! so the shortest route is the best one. It is small enough to train in a few seconds
//! and its optimal policy is known, which is what makes it a useful reference: after
//! training, a greedy walk should take 12 steps.

use athena::agent::DqnAgent;
use athena::network::InferenceBuffers;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::Array1;

const GRID: usize = 7;
const STATE_SIZE: usize = 2;
const ACTIONS: usize = 4;
const MAX_STEPS: usize = 60;

const GAMMA: f32 = 0.95;
const LEARNING_RATE: f32 = 0.002;
const BATCH_SIZE: usize = 64;
const EPISODES: usize = 400;

const SAVE_PATH: &str = "models/game_loop_dqn.bin";

/// Where the agent is. A game would have a lot more than this in it; what matters is
/// that `observe` turns it into a fixed-width vector of numbers on a similar scale.
struct World {
    x: usize,
    y: usize,
    steps: usize,
}

impl World {
    fn new() -> Self {
        World { x: 0, y: 0, steps: 0 }
    }

    /// The observation the agent sees. Both components are scaled to roughly 0 to 1: a
    /// network learns much faster when its inputs share a scale.
    fn observe(&self) -> Array1<f32> {
        Array1::from_vec(vec![
            self.x as f32 / (GRID - 1) as f32,
            self.y as f32 / (GRID - 1) as f32,
        ])
    }

    /// Returns the reward and whether the episode ended.
    fn step(&mut self, action: usize) -> (f32, bool) {
        match action {
            0 if self.x + 1 < GRID => self.x += 1,
            1 => self.x = self.x.saturating_sub(1),
            2 if self.y + 1 < GRID => self.y += 1,
            3 => self.y = self.y.saturating_sub(1),
            _ => {}
        }
        self.steps += 1;

        if self.x == GRID - 1 && self.y == GRID - 1 {
            (1.0, true)
        } else if self.steps >= MAX_STEPS {
            (-0.01, true)
        } else {
            (-0.01, false)
        }
    }
}

fn train() -> DqnAgent {
    // Adam, not SGD. A squared error on Q-values of this magnitude diverges under a
    // plain gradient step. Adam::new(&[], ..) grows its per-layer state on first use,
    // so it does not need the layers up front.
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));

    // Starts fully exploring; decay_epsilon below walks it down to 0.05.
    // The target network is refreshed every 200 training steps.
    let mut agent = DqnAgent::new_seeded(
        &[STATE_SIZE, 64, 64, ACTIONS],
        1.0,
        optimizer,
        200,
        true,
        7,
    );

    let mut buffer = ReplayBuffer::new(20_000);
    let mut rng = seeded_rng(11);

    for episode in 0..EPISODES {
        let mut world = World::new();

        loop {
            let state = world.observe();

            // One decision. This is the only call a frame has to make.
            let action = agent.act(state.view()).expect("state width matches the network");

            let (reward, done) = world.step(action);

            buffer.add(Experience {
                state,
                action,
                reward,
                next_state: world.observe(),
                done,
            });

            // Learning happens on the same thread here for simplicity. See
            // examples/background_training.rs for moving it off the frame thread.
            if buffer.len() >= BATCH_SIZE {
                let batch = buffer.sample_with(BATCH_SIZE, &mut rng);
                agent
                    .train_on_batch(&batch, GAMMA, LEARNING_RATE)
                    .expect("batch shapes match the network");
            }

            if done {
                break;
            }
        }

        agent.decay_epsilon(0.985, 0.05);

        if (episode + 1) % 100 == 0 {
            println!(
                "episode {:>3}  epsilon {:.3}  greedy walk {}",
                episode + 1,
                agent.epsilon,
                describe_walk(&agent)
            );
        }
    }

    agent
}

/// Steps a greedy walk takes, or a note that it never arrived.
fn describe_walk(agent: &DqnAgent) -> String {
    match greedy_steps(agent) {
        Some(steps) => format!("{} steps", steps),
        None => "did not reach the goal".to_string(),
    }
}

/// Walks the grid taking the best action every time, with no exploration.
///
/// Note this takes `&self` and calls `predict`, not `act`: `act` needs `&mut self`
/// because it draws from the agent's generator, and it can return a random action. An
/// evaluation that calls `act` measures the exploration schedule, not the policy.
fn greedy_steps(agent: &DqnAgent) -> Option<usize> {
    let mut world = World::new();
    let mut buffers = InferenceBuffers::new();

    for step in 0..MAX_STEPS {
        let state = world.observe();
        let q_values = agent.q_network.predict_into(state.view(), &mut buffers);

        let action = q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let (_, done) = world.step(action);
        if done {
            return if world.x == GRID - 1 && world.y == GRID - 1 {
                Some(step + 1)
            } else {
                None
            };
        }
    }

    None
}

fn main() {
    println!("training on a {}x{} grid, {} episodes", GRID, GRID, EPISODES);
    let agent = train();

    let trained = greedy_steps(&agent);
    println!("\ntrained greedy walk: {:?} steps, shortest possible is {}", trained, 2 * (GRID - 1));

    std::fs::create_dir_all("models").expect("could not create the models directory");
    agent.save(SAVE_PATH).expect("could not write the model");
    println!("saved to {}", SAVE_PATH);

    // A reloaded agent carries its epsilon, so set it to zero before evaluating.
    // The generator is not saved: a loaded agent explores from a fresh one.
    let mut loaded = DqnAgent::load(SAVE_PATH).expect("could not read the model back");
    loaded.update_epsilon(0.0);

    let reloaded = greedy_steps(&loaded);
    println!("reloaded greedy walk: {:?} steps", reloaded);
    assert_eq!(trained, reloaded, "the reloaded agent behaved differently");
}
