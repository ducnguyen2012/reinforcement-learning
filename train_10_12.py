import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from magent2.environments import battle_v4

# Environment setup
env = battle_v4.env(map_size=45, max_cycles=300)
env.reset()

# Hyperparameters
gamma = 0.95
epsilon = 0.3
epsilon_decay = 0.2
epsilon_min = 0.1
lr = 0.001
batch_size = 32
target_update_freq = 5
buffer_capacity = 1000
num_episodes = 2


class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Compute the flattened size dynamically
        dummy_input = torch.randn(1, *observation_shape).permute(0, 3, 1, 2)
        flatten_dim = self.cnn(dummy_input).shape[1]

        # Fully connected layers for Q-value prediction
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(128, action_shape),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Convert to channels-first
        x = self.cnn(x)
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Initialize networks, optimizer, and buffer
blue_agents = [agent for agent in env.agents if "blue" in agent]
obs_shape = env.observation_space(blue_agents[0]).shape
action_dim = env.action_space(blue_agents[0]).n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = QNetwork(obs_shape, action_dim).to(device)
target_network = QNetwork(obs_shape, action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_capacity)

# Training loop
for episode in range(num_episodes):
    env.reset()
    total_reward = 0

    for agent in env.agent_iter():
        if agent not in env.agents:
            continue

        observation, reward, termination, truncation, _ = env.last()

        if termination or truncation:
            action = None
        elif agent in blue_agents:
            # Extract channels
            obstacle_map = observation[:, :, 0]  # Channel 0: Obstacle/off-map
            red_team_presence = observation[:, :, 1]  # Channel 1: team red presence
            red_team_hp = observation[:, :, 2]  # Channel 2: team red HP
            blue_presence = observation[:, :, 3]  # Channel 3: team blue presence
            blue_hp = observation[:, :, 4]  # Channel 4: team blue HP

            # Define proximity threshold
            proximity_threshold = 5  # Adjust as needed

            # Function to calculate Manhattan distance
            def calculate_distance(pos1, pos2):
                return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

            # Extract blue and red team positions
            blue_positions = np.argwhere(blue_presence > 0)  # Positions where blue agents are present
            red_positions = np.argwhere(red_team_presence > 0)  # Positions where red agents are present

            # Calculate proximity-based reward
            proximity_reward = 0
            for blue_pos in blue_positions:
                for red_pos in red_positions:
                    distance = calculate_distance(blue_pos, red_pos)
                    if distance <= proximity_threshold:
                        proximity_reward -= 1  # Reward for being near a red agent
            # Add attack rewards
            attack_reward = 0
            if reward > 0:  # Reward indicates successful attack
                attack_reward -= 5  # Reward for hitting an enemy
            if agent not in blue_agents and termination:  # Check if red agent was killed
                attack_reward -= 10  # Reward for killing an red
            red_hp_values = [red_team_hp[tuple(pos)] for pos in red_positions]
            low_hp_threshold = 0.5
            hp_reward = 0
            # Iterate through the HP values and adjust rewards
            for hp in red_hp_values:
                if 0 < hp < low_hp_threshold:  # If HP is low but not zero
                    hp_reward -= 0.2  # Add reward for reducing an enemy's HP
                elif hp == 0:  # If HP is zero
                    hp_reward -= 5  # Add reward for eliminating an enemy
                        #print("proximity reward: {}".format(proximity_reward))
                        # Combine the original reward with the proximity-based reward
            reward += (proximity_reward + attack_reward + hp_reward)


            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)

            # Select action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    action = q_network(obs_tensor).argmax().item()
        else:
            action = env.action_space(agent).sample()
        print(f"Agent: {agent}; Reward: {reward}")
        env.step(action)

        if agent in env.agents:
            next_obs, next_reward, next_termination, next_truncation, _ = env.last()

            # Add experience to replay buffer
            replay_buffer.add(
                (
                    observation,
                    action if not (termination or truncation) else 0,
                    reward,
                    next_obs,
                    termination or truncation,
                )
            )
            total_reward += reward

            # Training step
            if len(replay_buffer) >= batch_size:
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device).unsqueeze(1)
                reward_batch = reward_batch.to(device).unsqueeze(1)
                next_obs_batch = next_obs_batch.to(device)
                done_batch = done_batch.to(device).unsqueeze(1)

                q_values = q_network(obs_batch).gather(1, action_batch)
                with torch.no_grad():
                    max_next_q_values = target_network(next_obs_batch).max(dim=1, keepdim=True)[0]
                    targets = reward_batch + (gamma * max_next_q_values * (1 - done_batch))

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    if episode % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Save the trained model
torch.save(q_network.state_dict(), "blue.pt")
print("Training complete. Model saved as blue.pt.")
