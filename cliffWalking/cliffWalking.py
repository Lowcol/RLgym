import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def epsilon_greedy_action(q_table, state, epsilon, env, rng):
    """Choose action using epsilon-greedy policy"""
    if rng.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

def q_learning(episodes, learning_rate=0.5, discount_factor=1, render=False):    
    env = gym.make("CliffWalking-v1", render_mode="human" if render else None)
    
    # Initialize Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # 48x4

    epsilon = 0.1
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)
    
    for episode in range(episodes):        
        state = env.reset()[0]
        terminated = False
        total_reward = 0

        while not terminated:  
            action = epsilon_greedy_action(q_table, state, epsilon, env, rng)
            
            next_state, reward, terminated, _, _ = env.step(action)
            
            # Q-Learning update (off-policy)
            # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            
            total_reward += reward
            state = next_state
        
        rewards_per_episode[episode] = total_reward
    
    env.close()
    
    # Save Q-table
    with open("cliffWalking/q_learning_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    
    return rewards_per_episode, q_table

def sarsa(episodes, learning_rate=0.5, discount_factor=1, render=False):    
    env = gym.make("CliffWalking-v1", render_mode="human" if render else None)
    
    # Initialize Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # 48x4
    
    epsilon = 0.1
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)
    
    for episode in range(episodes):        
        state = env.reset()[0]
        action = epsilon_greedy_action(q_table, state, epsilon, env, rng)
        terminated = False
        total_reward = 0
        
        while not terminated:
            next_state, reward, terminated, _, _ = env.step(action)
            
            # Choose next action using epsilon-greedy
            next_action = epsilon_greedy_action(q_table, next_state, epsilon, env, rng)
            
            # SARSA update (on-policy)
            # Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
            q_table[state, action] += learning_rate * (
                reward + discount_factor * q_table[next_state, next_action] - q_table[state, action]
            )

            
            total_reward += reward
            state = next_state
            action = next_action
        
        rewards_per_episode[episode] = total_reward
    
    env.close()
    
    # Save Q-table
    with open("cliffWalking/sarsa_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    
    return rewards_per_episode, q_table

def compare_algorithms(episodes=500):    
    # Run Q-Learning
    q_learning_rewards, q_learning_table = q_learning(episodes)
    
    # Run SARSA
    sarsa_rewards, sarsa_table = sarsa(episodes)
    
    plt.figure(figsize=(10, 6))
    plt.plot(smooth(q_learning_rewards), label='Q-learning')
    plt.plot(smooth(sarsa_rewards), label='SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs SARSA - CliffWalking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cliffWalking/cliffWalkingSARSAvsQL.png', dpi=300)
    plt.show()  
    
    return q_learning_rewards, sarsa_rewards, q_learning_table, sarsa_table

def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def test_trained_agent(algorithm='q_learning', render=True, episode=1):
    env = gym.make("CliffWalking-v1", render_mode="human" if render else None)
    
    # Load trained model
    if algorithm == 'q_learning':
        with open("cliffWalking/q_learning_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    else:
        with open("cliffWalking/sarsa_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    
    state, _ = env.reset()
    terminated = False
    total_reward = 0
    steps = 0
    
    while not terminated:
        action = np.argmax(q_table[state, :])  # Greedy action
        state, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        steps += 1   
    
    env.close()

if __name__ == "__main__":
    # Run comparison between Q-Learning and SARSA
    compare_algorithms(episodes=500)
    
    # Uncomment to test individual trained agents
    # test_trained_agent('q_learning', render=True)
    # test_trained_agent('sarsa', render=True)
    