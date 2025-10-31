import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import gymnasium as gym  

# env=gym.make("CartPole-v1")
# # print("Observation Space:", env.observation_space)

# observation, info = env.reset()
# print("observation:", observation)
# print("action space:", env.action_space)



# def basic_policy(obs):
#     angle = obs[2]
#     return 0 if angle < 0 else 1

# env = gym.make("CartPole-v1", render_mode="human")
# totals = []

# for episode in range(100):
#     print(f"Game :{episode}")
#     episode_reward = 0
#     obs = env.reset()[0]
#     for step in range(200):
#         action = basic_policy(obs)
#         obs, reward, done, state, info = env.step(action)
#         episode_reward += reward
#         if done:
#             break
#         print(f"Steps: {step}")
#     totals.append(episode_reward)
#     print(f"Episode {episode} reward: {episode_reward}")

# env.close()


# --- Policy Network ---
class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# --- Helper: Compute discounted rewards ---
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # normalize for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns

# --- Setup ---
env = gym.make("CartPole-v1", render_mode="human")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = PolicyNet(obs_size, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=0.01)


# --- Training Loop ---
for episode in range(500):
    obs = env.reset()[0]
    log_probs = []
    rewards = []
    
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        obs, reward, done, trunc, info = env.step(action.item())
        
        log_probs.append(log_prob)
        rewards.append(reward)
        
        if done:
            break

    # Compute returns and loss
    returns = compute_returns(rewards)
    loss = -(torch.stack(log_probs) * returns).sum()

    # Gradient step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    print(f"Episode {episode}: total reward = {total_reward}")

env.close()