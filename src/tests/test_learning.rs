//! End-to-end evidence that the algorithms learn.
//!
//! Every test here has a known optimal policy and asserts the agent reaches it, not that
//! a loss is finite or that a call does not panic. They run on tiny tasks so the whole
//! file stays under a second.
//!
//! The corridor is the reference task: if it fails, the fault is in the library rather
//! than in whichever example is being debugged.

use ndarray::{array, Array1};

use crate::agent::DqnAgent;
use crate::algorithms::{A2CAgent, A2CExperience, SACAgent, SACExperience, TD3Agent, TD3Experience};
use crate::optimizer::{Adam, OptimizerWrapper};
use crate::replay_buffer::{Experience, ReplayBuffer};
use crate::rng::seeded_rng;

/// A corridor of `LENGTH` cells. Action 1 moves right, action 0 moves left. Reaching the
/// right end pays 1.0 and ends the episode; every other step pays 0. The optimal policy
/// is to always move right, so a trained agent walks it in `LENGTH - 1` steps.
const LENGTH: usize = 6;

fn corridor_state(position: usize) -> Array1<f32> {
    let mut state = Array1::zeros(LENGTH);
    state[position] = 1.0;
    state
}

fn corridor_step(position: usize, action: usize) -> (usize, f32, bool) {
    let next = if action == 1 {
        (position + 1).min(LENGTH - 1)
    } else {
        position.saturating_sub(1)
    };

    if next == LENGTH - 1 {
        (next, 1.0, true)
    } else {
        (next, 0.0, false)
    }
}

/// Steps taken to walk the corridor greedily. `LENGTH - 1` is optimal, and the cap
/// reports failure rather than looping forever.
fn greedy_walk(agent: &mut DqnAgent) -> usize {
    let mut position = 0;
    for step in 0..(LENGTH * 4) {
        let q_values = agent.q_network.forward(corridor_state(position).view());
        let action = if q_values[1] > q_values[0] { 1 } else { 0 };
        let (next, _, done) = corridor_step(position, action);
        if done {
            return step + 1;
        }
        position = next;
    }
    usize::MAX
}

#[test]
fn dqn_learns_to_walk_a_corridor() {
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let mut agent = DqnAgent::new_seeded(&[LENGTH, 32, 2], 1.0, optimizer, 50, true, 4);
    let mut buffer = ReplayBuffer::new(2000);
    let mut rng = seeded_rng(9);

    for episode in 0..300 {
        // Anneal exploration, otherwise the greedy policy is never exercised
        agent.update_epsilon((1.0 - episode as f32 / 200.0).max(0.05));

        let mut position = 0;
        for _ in 0..(LENGTH * 3) {
            let state = corridor_state(position);
            let action = agent.act(state.view()).unwrap();
            let (next_position, reward, done) = corridor_step(position, action);

            buffer.add(Experience {
                state,
                action,
                reward,
                next_state: corridor_state(next_position),
                done,
            });

            if buffer.len() >= 32 {
                let batch = buffer.sample_with(32, &mut rng);
                agent.train_on_batch(&batch, 0.95, 0.005).unwrap();
            }

            position = next_position;
            if done {
                break;
            }
        }
    }

    let steps = greedy_walk(&mut agent);
    assert_eq!(
        steps,
        LENGTH - 1,
        "greedy policy took {} steps, optimal is {}",
        steps,
        LENGTH - 1
    );
}

#[test]
fn dqn_q_values_approach_the_discounted_return() {
    // From position p the optimal return is gamma^(distance to the end - 1)
    let gamma = 0.95f32;
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let mut agent = DqnAgent::new_seeded(&[LENGTH, 32, 2], 1.0, optimizer, 50, true, 11);
    let mut buffer = ReplayBuffer::new(2000);
    let mut rng = seeded_rng(13);

    for episode in 0..400 {
        agent.update_epsilon((1.0 - episode as f32 / 250.0).max(0.05));

        let mut position = 0;
        for _ in 0..(LENGTH * 3) {
            let state = corridor_state(position);
            let action = agent.act(state.view()).unwrap();
            let (next_position, reward, done) = corridor_step(position, action);

            buffer.add(Experience {
                state,
                action,
                reward,
                next_state: corridor_state(next_position),
                done,
            });

            if buffer.len() >= 32 {
                let batch = buffer.sample_with(32, &mut rng);
                agent.train_on_batch(&batch, gamma, 0.005).unwrap();
            }

            position = next_position;
            if done {
                break;
            }
        }
    }

    // One step from the end, moving right pays 1.0 immediately
    let near_end = agent.q_network.forward(corridor_state(LENGTH - 2).view());
    assert!(
        (near_end[1] - 1.0).abs() < 0.25,
        "Q(one from the end, right) was {}, expected about 1.0",
        near_end[1]
    );

    // Two steps out it is worth gamma, which must be visibly less
    let further = agent.q_network.forward(corridor_state(LENGTH - 3).view());
    assert!(
        further[1] < near_end[1],
        "Q should fall with distance: {} at two steps vs {} at one",
        further[1],
        near_end[1]
    );
}

#[test]
fn a2c_prefers_the_action_that_pays() {
    // One state, two actions, action 1 always pays. The policy has to concentrate on it.
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let mut agent = A2CAgent::new(2, 2, &[16], optimizer, 0.9, 5, 0.001, 0.5);
    agent.set_seed(3);

    let state = array![1.0, 0.0];

    for _ in 0..300 {
        let experiences: Vec<A2CExperience> = (0..5)
            .map(|step| {
                let action = step % 2;
                A2CExperience {
                    state: state.clone(),
                    action,
                    reward: if action == 1 { 1.0 } else { -1.0 },
                    next_state: state.clone(),
                    done: step == 4,
                    log_prob: 0.0,
                    value: agent.get_value(state.view()),
                }
            })
            .collect();

        agent.train(&experiences, 0.01).unwrap();
    }

    let probs = agent.get_action_probs(state.view());
    assert!(
        probs[1] > 0.7,
        "policy should concentrate on the paying action, got {:?}",
        probs
    );
}

#[test]
fn sac_critic_learns_a_known_q_function() {
    // Q(s, a) = a[0] with no dynamics: every transition terminates, so the target is the
    // reward and the critic has a fixed function to fit
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let mut agent = SACAgent::new(2, 1, &[32], optimizer, 0.99, 0.005, 0.2, false);
    agent.set_seed(5);

    let state = array![0.5, -0.5];
    let mut rng = seeded_rng(17);

    for _ in 0..200 {
        let experiences: Vec<SACExperience> = (0..16)
            .map(|_| {
                use rand::Rng;
                let a: f32 = rng.gen_range(-1.0..1.0);
                SACExperience {
                    state: state.clone(),
                    action: array![a],
                    reward: a,
                    next_state: state.clone(),
                    done: true,
                }
            })
            .collect();

        agent.update(&experiences, 0.005).unwrap();
    }

    let high = agent.get_q_value(state.view(), array![0.9].view()).0;
    let low = agent.get_q_value(state.view(), array![-0.9].view()).0;

    assert!(
        high > low,
        "critic should rank a=0.9 above a=-0.9, got {} and {}",
        high,
        low
    );
}

#[test]
fn td3_critic_learns_a_known_q_function() {
    let optimizer = OptimizerWrapper::Adam(Adam::new(&[], 0.9, 0.999, 1e-8));
    let mut agent = TD3Agent::new(2, 1, &[32], optimizer, 0.99, 0.005, 2, -1.0, 1.0);
    agent.set_seed(23);

    let state = array![0.5, -0.5];
    let mut rng = seeded_rng(29);

    for _ in 0..200 {
        let experiences: Vec<TD3Experience> = (0..16)
            .map(|_| {
                use rand::Rng;
                let a: f32 = rng.gen_range(-1.0..1.0);
                TD3Experience {
                    state: state.clone(),
                    action: array![a],
                    reward: a,
                    next_state: state.clone(),
                    done: true,
                }
            })
            .collect();

        agent.update(&experiences, 0.005, 0.005).unwrap();
    }

    let high = agent.get_q_values(state.view(), array![0.9].view()).0;
    let low = agent.get_q_values(state.view(), array![-0.9].view()).0;

    assert!(
        high > low,
        "critic should rank a=0.9 above a=-0.9, got {} and {}",
        high,
        low
    );
}
