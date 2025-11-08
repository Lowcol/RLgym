import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="human" if render else None)

    # Policy network
    policy = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
        nn.Softmax(dim=-1)
    )

    if is_training:
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    else:
        with open("cartpole/cartpole.pkl", "rb") as f:
            policy.load_state_dict(pickle.load(f))

    gamma = 0.99
    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(action.item())

            if is_training:
                log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            episode_reward += reward
            state = next_state

        rewards_per_episode[episode] = episode_reward

        if is_training:
            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy gradient update
            log_probs = torch.stack(log_probs)
            loss = -(log_probs * returns.detach()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.1f}")

        # Clean up environment to prevent slowdowns
        env.close()
        env = gym.make(env_name, render_mode="human" if render else None)

    # Save model
    if is_training:
        with open("cartpole/cartpole.pkl", "wb") as f:
            pickle.dump(policy.state_dict(), f)

    # Plot running mean reward
    mean_reward = np.zeros(episodes)
    for t in range(episodes):
        mean_reward[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_reward)
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.title("REINFORCE on CartPole")
    plt.savefig("cartpole.png")
    plt.close()

    env.close()

if __name__ == "__main__":
    run(200, is_training=False, render=True)
