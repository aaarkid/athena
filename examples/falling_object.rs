/***
# Catch the falling object.

* The game state is 2D: the position of the player (1D) and the position of the falling object (1D).
* There are three possible actions: move left, stay in place, and move right.
* The object moves left every turn.
* If the player is under the object at the end of step, the player gets a reward of +1.
* After the player catches the object, player goes to left-most position and the object appears in the 3 right-most positions.
***/

// Import the necessary modules from the athena library.
use athena::{
    agent::DqnAgent,
    replay_buffer::{ReplayBuffer, Experience},
    optimizer::{OptimizerWrapper, SGD},
};
use ndarray::Array1;

// These constants are parameters of the game, the agent and the training process.
const BOARD_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.001;
const EPSILON: f32 = 0.1;
const GAMMA: f32 = 0.9;
const CAPACITY: usize = 10000;
const BATCH_SIZE: usize = 32;

// Function to initialize the state of the game, which represents the positions of the player and the object.
fn initialize_state(player_pos: usize, object_pos: usize) -> Array1<f32> {
    Array1::from(vec![player_pos as f32 / BOARD_SIZE as f32, object_pos as f32 / BOARD_SIZE as f32])
}

// Function that determines the reward and the end condition of the game.
fn game_logic(player_pos: usize, object_pos: usize) -> (f32, bool) {
    if player_pos == object_pos {
        (1.0, true)  // Positive reward and end the game if player reaches the object.
    } else {
        (-0.5, false)  // Negative reward otherwise, and the game continues.
    }
}

fn main() {
    let now = std::time::Instant::now();
    // Creating the Deep Q-Network (DQN) agent.
    let layer_sizes = &[2, 10, 3];
    // let layers = Layer::to_vector(layer_sizes);
    let optimizer = OptimizerWrapper::SGD(SGD::new());
    let mut agent = DqnAgent::new(layer_sizes, EPSILON, optimizer);

    // Creating a replay buffer to store the experiences for training.
    let mut replay_buffer = ReplayBuffer::new(CAPACITY);

    // Initializing the game state.
    let mut player_pos = 0;
    let mut object_pos = BOARD_SIZE / 2;
    let mut state = initialize_state(player_pos, object_pos);
    
    for _step in 0..50 {
        // This variable will be used to compute the total reward.
        let mut total_reward = 0.0;

        // Loop over episodes.
        for _episode in 0..100 {
            // The agent chooses an action based on the current state.
            let action = agent.act(state.view());

            // Update the player's position based on the action.
            match action {
                0 => {
                    player_pos = (player_pos as isize - 1).rem_euclid(BOARD_SIZE as isize) as usize;
                },
                1 => {},
                2 => {
                    player_pos = (player_pos + 1).rem_euclid(BOARD_SIZE);
                },
                _ => panic!("Invalid action"),
            }

            // Determine the reward and whether the game is done based on the updated state.
            let (reward, done) = game_logic(player_pos, object_pos);

            // Add the reward to the total reward.
            total_reward += reward;

            // Generate the new state after taking the action.
            let new_state = initialize_state(player_pos, object_pos);

            // Construct an experience and add it to the replay buffer.
            let experience = Experience { state, next_state: new_state.clone(), action, reward, done };
            replay_buffer.add(experience);

            // Reset the game if it is done, otherwise update the state.
            if done {
                player_pos = 0;
                object_pos = (BOARD_SIZE - 3) + rand::random::<usize>() % 3;
                state = initialize_state(player_pos, object_pos);
            } else {
                object_pos = (object_pos as isize - 1).rem_euclid(BOARD_SIZE as isize) as usize;
                state = new_state;
            }

            // If there are enough experiences in the buffer, sample a batch and train the agent.
            if replay_buffer.len() > BATCH_SIZE {
                let experiences = replay_buffer.sample(BATCH_SIZE);
                agent.train_on_batch(&experiences, GAMMA, LEARNING_RATE);
            }
        }
        // Print the total reward and the time taken for training.
        println!("Total reward: {}", total_reward);
    }
    println!("Time taken: {:?}", now.elapsed());
}
