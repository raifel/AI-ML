
# Q-learning Agent Documentation

## Introduction

This Python script demonstrates the implementation of a Q-learning algorithm to train an agent in OpenAI's `Taxi-v3` environment. The goal is to enable the taxi agent to learn how to efficiently pick up and drop off passengers by maximizing cumulative rewards over time.

---

## Table of Contents

1. [Dependencies](#dependencies)  
2. [Reinforcement Learning Overview](#reinforcement-learning-overview)  
   - [Core RL Concepts](#core-rl-concepts)  
   - [Q-learning in RL](#q-learning-in-rl)  
3. [Code Overview](#code-overview)  
   - [Environment Initialization](#environment-initialization)  
   - [Q-learning Algorithm](#q-learning-algorithm)  
   - [Trip Animation](#trip-animation)  
4. [Parameters](#parameters)  
5. [Execution Instructions](#execution-instructions)  
6. [Potential Enhancements](#potential-enhancements)  

---

## Dependencies

Ensure the following Python libraries are installed:

- `gym` (OpenAI Gym toolkit for reinforcement learning)  
- `numpy` (for numerical operations)  

To install missing libraries, run:

```bash
pip install gym numpy
```

---

## Reinforcement Learning Overview

### Core RL Concepts

Reinforcement Learning (RL) is a branch of machine learning focused on training agents to make a sequence of decisions by interacting with an environment. The agent learns to maximize cumulative rewards by exploring and exploiting the environment.

Key components of RL include:

- **Agent**: The learner or decision-maker.  
- **Environment**: The external system with which the agent interacts.  
- **State (S)**: A representation of the current situation in the environment.  
- **Action (A)**: A set of possible moves the agent can take.  
- **Reward (R)**: Feedback from the environment after the agent performs an action.  
- **Policy (π)**: A strategy that the agent follows to decide actions based on states.  
- **Value Function (V)**: The expected long-term return from a state under a specific policy.  
- **Q-value (Q)**: The expected return for taking a specific action in a specific state.  

### Q-learning in RL

Q-learning is a model-free, off-policy RL algorithm that learns the optimal action-value function \( Q(s, a) \) by iteratively updating the Q-table based on the Bellman equation:

```
Q(s, a) ← (1 - α) Q(s, a) + α [ r + γ max_a' Q(s', a') ]
```

Where:  

- **α (learning rate)**: Controls how much new information overrides old information.  
- **γ (discount factor)**: Weighs the importance of future rewards.  
- **s, s'**: Current and next states.  
- **a, a'**: Current and next actions.  
- **r**: Immediate reward for the current state-action pair.  

---

## Code Overview

### Environment Initialization

The `Taxi-v3` environment simulates a grid-based map where:

- The taxi must pick up a passenger from a designated location.  
- The taxi must drop off the passenger at the correct destination.  

Key steps:

- `streets = gym.make("Taxi-v3", render_mode='ansi')` initializes the environment in textual rendering mode.  
- `streets.reset()` resets the environment to a random initial state.  
- `streets.encode(row, col, passenger_location, destination)` encodes a specific state, setting the taxi's position, passenger, and destination.  

### Q-learning Algorithm

The agent learns through interaction with the environment by updating a Q-table based on observed rewards.

**Key Components**:

- **Q-table Initialization**:  
  ```python
  q_table = np.zeros((streets.observation_space.n, streets.action_space.n))
  ```  
  The Q-table stores the expected rewards for all state-action pairs.

- **Training Loop**:
  - The agent takes an action (`explore` or `exploit`).  
  - Observes the reward and next state.  
  - Updates the Q-value using the Bellman equation.  

- **Exploration vs. Exploitation**:
  - To balance learning and performance, the agent randomly explores the environment with probability \( ε \) or exploits the best-known action.

### Trip Animation

After training, the script animates 5 trips to showcase the agent's learned behavior:

- The taxi starts at a random initial state.  
- It performs the optimal actions (based on the trained Q-table) until it completes the trip.  
- The environment state is rendered step-by-step in the console, creating an animation effect.

---

## Parameters

| Parameter          | Description                                             | Default Value |
|--------------------|---------------------------------------------------------|---------------|
| `learning_rate`    | Rate at which the Q-values are updated                  | 0.1           |
| `discount_factor`  | Importance of future rewards                            | 0.6           |
| `exploration`      | Probability of taking a random action (exploration)     | 0.1           |
| `epochs`           | Number of training iterations                           | 10,000        |
| `trip_length`      | Maximum steps for each animated trip                    | 25            |

---

## Execution Instructions

1. **Run the script**:  
   Execute the script in any Python environment:

   ```bash
   python taxi_qlearning.py
   ```

2. **View the animation**:  

   - Each trip will be displayed step-by-step in the console.  
   - The console clears between steps to simulate movement.  

---

## Potential Enhancements

1. **Hyperparameter Optimization**:
   - Adjust `learning_rate`, `discount_factor`, and `exploration` to improve learning performance.

2. **Performance Visualization**:
   - Plot the agent's total rewards per epoch to monitor training progress.

3. **Saving the Q-table**:
   - Save the trained Q-table to a file for future use:

     ```python
     np.save("q_table.npy", q_table)
     ```

4. **Dynamic Exploration**:
   - Implement an exploration decay strategy to reduce exploration over time:

     ```python
     exploration = max(0.01, exploration * 0.99)
     ```

5. **Advanced RL Techniques**:
   - Implement **Deep Q-Networks (DQN)** for more complex environments with larger state spaces.  
   - Experiment with **policy-based methods** such as Proximal Policy Optimization (PPO) or Actor-Critic.

---

## Conclusion

This script demonstrates a simple yet powerful implementation of Q-learning, a foundational RL algorithm, in a discrete state-action environment. By expanding the scope with advanced RL techniques, the agent can be adapted for more complex real-world problems.
