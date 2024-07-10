import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import mujoco as mj
from mujoco.glfw import glfw
import time
import matplotlib.pyplot as plt
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BUFFER_SIZE = 1000000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
ALPHA = 0.2  # Temperature parameter for entropy

# MuJoCo setup
xml_path = "car1.xml"  # Path to your MuJoCo model file
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Set initial state
data.qpos[0] = 0  # pos x
data.qpos[1] = 0  # pos y
data.qpos[2] = 0  # pos z
data.qpos[3] = 0  #
data.qpos[4] = 0  # roll
data.qpos[5] = 0  # pitch
data.qpos[6] = 0  # yaw
data.qpos[7] = 0  #
data.qpos[8] = 0  #

# Neural Network for Actor and Critic


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        )

        # Critic network (2 Q-functions)
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        mean, log_std = self.actor(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def critique(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.critic1(sa), self.critic2(sa)

# SAC Agent


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.actor_critic = ActorCritic(
            state_dim, action_dim, hidden_dim).to(device)
        self.actor_critic_target = ActorCritic(
            state_dim, action_dim, hidden_dim).to(device)
        self.actor_critic_target.load_state_dict(
            self.actor_critic.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor_critic.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(list(self.actor_critic.critic1.parameters()) +
                                           list(self.actor_critic.critic2.parameters()), lr=LEARNING_RATE)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.action_dim = action_dim
        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, log_std = self.actor_critic(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
        return action.cpu().numpy()[0]

    def train(self, batch):
        state, action, reward, next_state, done = [b.to(device) for b in batch]

        # Compute target Q-value
        with torch.no_grad():
            next_mean, next_log_std = self.actor_critic_target(next_state)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_z = next_normal.rsample()
            next_action = torch.tanh(next_z)
            next_log_pi = next_normal.log_prob(
                next_z) - torch.log(1 - next_action.pow(2) + 1e-6)
            next_log_pi = next_log_pi.sum(dim=-1, keepdim=True)

            target_q1, target_q2 = self.actor_critic_target.critique(
                next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_pi
            target_q = reward + (1 - done) * GAMMA * target_q

        # Compute current Q-value
        current_q1, current_q2 = self.actor_critic.critique(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + \
            F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        mean, log_std = self.actor_critic(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        q1, q2 = self.actor_critic.critique(state, action)
        q = torch.min(q1, q2)

        actor_loss = (ALPHA * log_pi - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target network
        for param, target_param in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
            target_param.data.copy_(
                TAU * param.data + (1 - TAU) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def save_checkpoint(self, episode, filename):
        torch.save({
            'episode': episode,
            'total_steps': self.total_steps,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_critic_target_state_dict': self.actor_critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
        }, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.actor_critic.load_state_dict(
                checkpoint['actor_critic_state_dict'])
            self.actor_critic_target.load_state_dict(
                checkpoint['actor_critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])
            self.replay_buffer = checkpoint['replay_buffer']
            self.total_steps = checkpoint['total_steps']
            print(f"Checkpoint loaded from {filename}")
            return checkpoint['episode']
        else:
            print(f"No checkpoint found at {filename}")
            return 0

# MuJoCo Environment wrapper


class MuJoCoEnv:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.target = np.array([2, 2])  # 2x2 square target
        self.max_steps = 1000

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        # Apply action
        self.data.ctrl[0] = action[0]  # Right wheel torque
        self.data.ctrl[1] = action[1]  # Left wheel torque

        # Step the simulation
        mj.mj_step(self.model, self.data)

        # Get new state
        next_state = self._get_obs()

        # Calculate reward
        reward = self._get_reward(next_state)

        # Check if done
        done = self._is_done(next_state)

        return next_state, reward, done, {}

    def _get_obs(self):
        return np.array([
            self.data.qpos[0],  # x position
            self.data.qpos[1],  # y position
            self.data.qpos[6],  # yaw
            self.data.qvel[0],  # x velocity
            self.data.qvel[5],  # angular velocity
        ])

    def _get_reward(self, state):
        distance_to_target = np.linalg.norm(state[:2] - self.target)
        return -distance_to_target  # Negative distance as reward

    def _is_done(self, state):
        distance_to_target = np.linalg.norm(state[:2] - self.target)
        return distance_to_target < 0.1 or self.data.time > self.max_steps * self.model.opt.timestep

# Visualization function


def plot_results(episode_rewards, avg_rewards):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('Episode Reward')
    plt.plot(episode_rewards)
    plt.subplot(122)
    plt.title('Average Reward')
    plt.plot(avg_rewards)
    plt.show()

# Training loop


def train(num_episodes=1000, save_interval=100, resume_from=None):
    env = MuJoCoEnv(model, data)
    agent = SACAgent(state_dim=5, action_dim=2, hidden_dim=256)

    episode_rewards = []
    avg_rewards = []

    start_episode = 0
    if resume_from:
        start_episode = agent.load_checkpoint(resume_from)

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        episode_reward = 0

        # Print the current episode
        print(f"Starting episode {episode + 1}/{num_episodes}")

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Replace agent.update() with:
            if len(agent.replay_buffer) > BATCH_SIZE:
                batch = agent.sample_batch()
                agent.train(batch)

            state = next_state
            episode_reward += reward

            # Print the episode reward
            print(f"Episode {episode + 1} reward: {episode_reward}")

            if (episode + 1) % save_interval == 0:
                agent.save_checkpoint(episode + 1, f"checkpoint_{episode + 1}.pth")
                print(f"Checkpoint saved at episode {episode + 1}")

    plot_results(episode_rewards, avg_rewards)
    print("Training completed.")

    # Save the final model
    agent.save_checkpoint(num_episodes, "sac_agent_final.pth")


if __name__ == "__main__":
    train()
