def calculate_formation_positions(blue_positions, formation="u", map_size=45):
    if formation == "line":
        target_y = map_size // 2  # Center vertically
        return [(pos[0], target_y) for pos in blue_positions]
    elif formation == "wedge":
        mid_x = map_size // 2
        return [(mid_x + i, map_size // 2 + i) for i, _ in enumerate(blue_positions)]
    elif formation == "square":
        return [(map_size // 4 + i, map_size // 4 + i) for i, _ in enumerate(blue_positions)]
    elif formation == "u":
        # In a U-formation, the top part will be a line, and the bottom will be an arc
        mid_x = map_size // 2
        bottom_y = map_size - 2  # Bottom part of the U
        top_y = map_size // 4  # Top part of the U
        width = len(blue_positions) // 2  # Number of agents for each side of the U

        # Generate positions for the U formation
        positions = []
        # Create the bottom horizontal line
        for i in range(-width, width + 1):
            positions.append((mid_x + i, bottom_y))
        # Create the vertical sides of the U
        for i in range(width):
            positions.append((mid_x - width, top_y + i))  # Left side
            positions.append((mid_x + width, top_y + i))  # Right side
        return positions[:len(blue_positions)]  # Ensure we return only the required number of positions


#! update training loop
# Training loop
for episode in range(num_episodes):
    env.reset()
    total_reward = 0
    step = 0
    for agent in env.agent_iter():
        step += 1
        if agent not in env.agents:
            continue

        observation, reward, termination, truncation, _ = env.last()
        obstacle_map = observation[:, :, 0]  # Channel 0: Obstacle/off-map
        red_team_presence = observation[:, :, 1]  # Channel 1: team red presence
        red_team_hp = observation[:, :, 2]  # Channel 2: team red HP
        blue_presence = observation[:, :, 3]  # Channel 3: team blue presence
        blue_hp = observation[:, :, 4]  # Channel 4: team blue HP
        blue_positions = np.argwhere(blue_presence > 0)  # Positions where blue agents are present
        red_positions = np.argwhere(red_team_presence > 0)  # Positions where red agents are present

        if termination or truncation:
            action = None
        elif agent in blue_agents:
            # Get positions for blue agents in a U-formation
            u_formation_positions = calculate_formation_positions(blue_positions, formation="u", map_size=45)

            # Move blue agents towards their U-formation positions
            target_pos = u_formation_positions[blue_agents.index(agent)]
            action_mask = action_mask(observation, blue_positions[blue_agents.index(agent)], target_pos)
            
            # Select action based on mask
            action = np.random.choice([0, 1, 2, 3, 4], p=action_mask)
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
