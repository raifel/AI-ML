import gym
import random
import numpy as np
from time import sleep
import os

# Set random seed for reproducibility
random.seed(1234)
np.random.seed(1234)

# Initialize the Taxi environment
streets = gym.make("Taxi-v3", render_mode='ansi')
initial_state_info = streets.reset()
initial_state = streets.encode(2, 3, 2, 0)

# Set the initial state
streets.s = initial_state

# Define the Q-learning parameters
q_table = np.zeros((streets.observation_space.n, streets.action_space.n))
learning_rate = 0.1
discount_factor = 0.6
exploration = 0.1
epochs = 10000

# Q-learning algorithm
for taxi_run in range(epochs):
    state_info = streets.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info
    done = False

    while not done:
        if random.uniform(0, 1) < exploration:
            action = streets.action_space.sample()  # Explore: choose a random action
        else:
            action = np.argmax(q_table[state])  # Exploit: choose the best known action

        next_state, reward, done, _, _ = streets.step(action)

        # Update Q-value
        next_max_q = np.max(q_table[next_state])
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] +                                  learning_rate * (reward + discount_factor * next_max_q)

        state = next_state  # Move to the next state

# Animate the trips
for tripnum in range(1, 6):  # Display 5 animated trips
    state_info = streets.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info
    done = False
    trip_length = 0

    print(f"Starting Trip {tripnum}...\n")
    while not done and trip_length < 25:
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console
        print(f"Trip {tripnum} - Step {trip_length}")
        print(streets.render())  # Render the current state
        sleep(0.5)  # Delay for animation effect

        action = np.argmax(q_table[state])  # Choose the best action
        next_state, reward, done, _, _ = streets.step(action)  # Take the action
        state = next_state  # Move to the next state
        trip_length += 1

    # Pause after completing the trip
    sleep(2)
