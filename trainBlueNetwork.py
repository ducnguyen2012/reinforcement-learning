from magent2.environments import battle_v4
import os
import cv2
import torch
env = battle_v4.env(map_size=45, render_mode="rgb_array")
print("This is my env after reset: {}".format(env.reset()))

from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch_model import QNetwork
# Hyperparameters
learning_rate = 1e-4
gamma = 0.99  # Discount factor
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 500  # Decay over 500 steps
batch_size = 64
buffer_capacity = 100000
target_update_freq = 100  # Update target network every 100 steps

# Initialize networks
policy_net = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n)
target_net = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n)
target_net.load_state_dict(policy_net.state_dict())  # Copy weights
target_net.eval()  # Target network in evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Replay buffer
replay_buffer = ReplayBuffer(buffer_capacity)
#print("This is my env.observation_space: {}".format(env.observation_space("blue_0")))
#print("This is env.action_space: {}".format(env.action_space("blue_0").n))
def epsilon_greedy_policy(state, epsilon):
    if random.random() < epsilon:
        return env.action_space("blue_0").sample()  # Random action
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax(dim=1).item()  # Exploit best action
num_episodes = 50  # Total number of episodes
epsilon = epsilon_start

for episode in range(num_episodes):
    
    state = env.reset()
    #print("This is my state: {}".format(state))
    observation, reward, termination, truncation, info = env.last()
    
    # Ensure 'observation' is valid
    if observation is None:
        print("Observation is None!")
        exit()
    else:
        #print("observation is not none!")
        state = {"blue_0": observation}  # Create a dictionary for your policy
    #print("This is state after reset: {}".format(state))
    total_reward = 0

    for step in range(100):  # Max steps per episode
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            # Select action using epsilon-greedy policy
            action = epsilon_greedy_policy(state["blue_0"], epsilon)
            #print(f"Step {step}: Reward = {reward}")
            env.step(action)
            #print(f"Agent: {agent}, Reward: {reward}")
            # Perform action in the environment
            next_state, reward, termination, truncation, _ = env.last()
            done = termination or truncation
            next_state = {"blue_0": next_state}
            #print("This is my next_state: {}".format(next_state))
            
            # Add experience to replay buffer
            replay_buffer.add((state["blue_0"], action, reward, next_state["blue_0"], done))

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            # Sample a minibatch and train the network
            if len(replay_buffer) >= batch_size:
                # Sample experiences
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Compute targets
                with torch.no_grad():
                    target_q_values = target_net(next_states.permute(0, 3, 1, 2)).max(1)[0]
                    targets = rewards + (1 - dones) * gamma * target_q_values

                # Compute current Q values
                q_values = policy_net(states.permute(0, 3, 1, 2))
                current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute loss
                loss = loss_fn(current_q_values, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update the target network periodically
            if step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
torch.save(policy_net.state_dict(), "blue.pt")
print("DONE SAVE BLUE.PT!")
