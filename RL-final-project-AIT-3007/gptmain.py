from magent2.environments import battle_v4
import os
import cv2
import torch
from torch_model import QNetwork

def get_action(agent, observation, q_network):
    # Assuming observation is in the form expected by your model (e.g., [C, H, W])
    observation = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(observation)
    action = torch.argmax(q_values, dim=1).numpy()[0]
    return action

if __name__ == "__main__":
    # Initialize environment
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35
    frames = []

    # Load the pretrained models for both red and blue
    red_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n)
    red_network.load_state_dict(torch.load("red.pt", weights_only=True, map_location="cpu"))

    blue_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n)
    blue_network.load_state_dict(torch.load("blue.pt", weights_only=True, map_location="cpu"))

    # Reset the environment for a new episode
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # This agent has died
        else:
            if agent == "red_0":
                # Get action from the red agent's model (red.pt)
                action = get_action(agent, observation, red_network)
            elif agent == "blue_0":
                # Get action from the blue agent's model (blue.pt)
                action = get_action(agent, observation, blue_network)
            else:
                # Other agents can take random actions
                action = env.action_space(agent).sample()

        # Perform the action in the environment
        env.step(action)

        # Capture frames for red_0 and blue_0 to create a video
        if agent == "red_0" or agent == "blue_0":
            frames.append(env.render())

    # Save the frames as a video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"blue_vs_red.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    print("Done recording the fight between red_0 and blue_0")

    # Close the environment
    env.close()
