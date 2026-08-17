//! Training on a worker thread while a game loop keeps running.
//!
//! The pattern a game needs: the main thread steps the world and picks actions at frame
//! rate, a worker owns the agent and the replay buffer, and the two talk over channels.
//! The main thread sends experiences and receives a fresh copy of the network whenever
//! the worker has trained enough to be worth swapping in.
//!
//! This compiles only because the agents hold an `StdRng`. `ThreadRng` wraps an `Rc`, so
//! an agent holding one cannot be moved onto another thread at all.

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use athena::agent::DqnAgent;
use athena::network::NeuralNetwork;
use athena::optimizer::{Adam, OptimizerWrapper};
use athena::replay_buffer::{Experience, ReplayBuffer};
use athena::rng::seeded_rng;
use ndarray::{array, Array1};

const STATE_SIZE: usize = 4;
const ACTIONS: usize = 2;
const FRAMES: usize = 1500;
/// A compressed frame. A real one is 16.7 ms at 60 fps; the point is only that the frame
/// hands the worker some wall clock, which is what keeps the actor's network fresh.
const FRAME_BUDGET: std::time::Duration = std::time::Duration::from_millis(1);
const BATCH_SIZE: usize = 32;
const SEED: u64 = 20260817;

/// A stand-in for the game: the reward says whether the action matched the sign of the
/// first state component, which is learnable in a few thousand steps.
struct Toy {
    state: Array1<f32>,
    step: usize,
}

impl Toy {
    fn new() -> Self {
        Toy { state: Array1::zeros(STATE_SIZE), step: 0 }
    }

    fn advance(&mut self, action: usize) -> (Array1<f32>, f32, bool) {
        let t = self.step as f32 * 0.05;
        let next = array![t.sin(), t.cos(), (t * 0.3).sin(), 0.0];

        let wants_right = self.state[0] > 0.0;
        let reward = if (action == 1) == wants_right { 1.0 } else { -1.0 };

        self.step += 1;
        let done = self.step % 50 == 0;
        self.state = next.clone();
        (next, reward, done)
    }
}

/// What the worker sends back: a trained network to swap into the actor.
enum FromWorker {
    Weights(Box<NeuralNetwork>),
}

fn main() {
    println!("Background training");
    println!("===================\n");

    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let learner = DqnAgent::new_seeded(
        &[STATE_SIZE, 64, 64, ACTIONS],
        0.1,
        optimizer,
        200,
        true,
        SEED,
    );

    // The actor is a copy of the learner's network. It only ever runs forward.
    let mut actor = learner.q_network.clone();

    let (to_worker, from_main) = mpsc::channel::<Experience>();
    let (to_main, from_worker) = mpsc::channel::<FromWorker>();

    // The whole agent moves onto the worker. Nothing is shared, so nothing is locked.
    let worker = thread::spawn(move || {
        let mut agent = learner;
        let mut buffer = ReplayBuffer::new(10_000);
        let mut rng = seeded_rng(SEED ^ 0xa5a5);
        let mut updates = 0usize;

        // Ends when the main thread drops its sender
        while let Ok(experience) = from_main.recv() {
            buffer.add(experience);

            if buffer.len() < BATCH_SIZE {
                continue;
            }

            let batch = buffer.sample_with(BATCH_SIZE, &mut rng);
            let _ = agent.train_on_batch(&batch, 0.99, 0.001);
            updates += 1;

            // Hand the actor a fresh network every so often. Cloning a small network is
            // cheap next to the training step that produced it.
            if updates % 25 == 0
                && to_main
                    .send(FromWorker::Weights(Box::new(agent.q_network.clone())))
                    .is_err()
            {
                break;
            }
        }

        (agent, updates)
    });

    let mut env = Toy::new();
    let mut state = env.state.clone();
    let mut swaps = 0usize;
    let mut recent_reward = 0.0;
    let mut window = Vec::new();
    let started = Instant::now();

    for frame in 0..FRAMES {
        // Frame work: one forward pass, greedy. No training, no locking, no allocation
        // beyond the forward pass itself.
        let q_values = actor.forward(state.view());
        let action = if frame < 200 {
            frame % ACTIONS // warm up with something other than a fixed policy
        } else {
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        let (next_state, reward, done) = env.advance(action);
        recent_reward += reward;
        window.push(reward);

        // Hand the transition to the worker and move on. Send never blocks.
        if to_worker
            .send(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            })
            .is_err()
        {
            break;
        }

        // Pick up a newer network if one is waiting. try_recv never blocks the frame.
        while let Ok(FromWorker::Weights(network)) = from_worker.try_recv() {
            actor = *network;
            swaps += 1;
        }

        state = next_state;

        // Stand in for the rest of the frame. Without it the main thread outruns the
        // worker, the channel backs up, and the actor never sees an updated network.
        thread::sleep(FRAME_BUDGET);

        if (frame + 1) % 250 == 0 {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            println!(
                "frame {:>5}   mean reward {:>6.3}   network swaps {}",
                frame + 1,
                mean,
                swaps
            );
            window.clear();
        }
    }

    // Dropping the sender ends the worker's recv loop
    drop(to_worker);
    let (_agent, updates) = worker.join().expect("the training thread panicked");

    println!("\n{} frames in {:.2}s", FRAMES, started.elapsed().as_secs_f32());
    println!("{} training steps ran on the worker", updates);
    println!("{} networks swapped into the actor", swaps);
    println!("total reward {:.0}", recent_reward);
}
