import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 1. The Network (Actor and Critic share the input layer)
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Actor Head (Outputs probabilities of actions)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic Head (Outputs a single Value scalar)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # Actor: Softmax to get probabilities (e.g., [0.8, 0.2])
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: Just a raw number (e.g., "This state is worth 10 points")
        state_value = self.critic(x)
        
        return action_probs, state_value

# Hyperparameters
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n              # 3 (forward, backward, do nothing)
LEARNING_RATE = 0.01
GAMMA = 0.99  # Discount factor for future rewards

model = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    scores = []
    
    for episode in range(1000):
        state, _ = env.reset()
        done = False
        score = 0
        
        # Storage for this episode
        log_probs = []
        values = []
        rewards = []

        # --- COLLECT TRAJECTORY ---
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get Action and Value predictions
            probs, value = model(state_tensor)
            
            # Sample action from probability distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store data
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            score += reward
            state = next_state

        # --- UPDATE NETWORK ---
        # 1. Calculate Discounted Returns (moving backwards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Normalize for stability
        
        # 2. Calculate Losses
        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)
        
        # Advantage = Actual Return - Predicted Value
        advantage = returns - values
        
        # Actor Loss: -log_prob * advantage
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # Critic Loss: MSE(value, return)
        critic_loss = F.mse_loss(values, returns)
        
        # Total Loss
        loss = actor_loss + critic_loss
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        scores.append(score)
        avg_score = np.mean(scores[-50:]) # Average of last 50 runs
        
        if episode % 20 == 0:
            print(f"Episode {episode} | Score: {score:.0f} | Avg Score: {avg_score:.1f}")
        
        # CartPole is considered solved at 475+ usually (500 max)
        if avg_score > 490:
            print(f"Solved in {episode} episodes!")
            break

train()