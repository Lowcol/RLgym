import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human" if render else None)

    
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 8x8 world with 4 actions
    else:
        f = open("FrozenLake/frozen_lake.pkl", "rb")
        q = pickle.load(f)
        f.close()
        
        
    learning_rate_a = 0.9 #alpha or learning rate
    discount_factor_g = 0.9 #gamma or discount factor
    epsilons = 1 # 1 = 100% random action
    epsilon_decay_rate = 0.0001 # decay per episode
    rng = np.random.default_rng() # random number generator
    
    rewards_per_episode = np.zeros(episodes)
    
    
    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}")
        state = env.reset()[0]
        terminated = False
        truncated = False


        while(not terminated and not truncated):
            if is_training and rng.random() < epsilons:
                #choose random action (0=move left, 1=move down, 2=move right, 3=move up)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if is_training:
                #update q value
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state
            
        
        epsilons = max(0, epsilons - epsilon_decay_rate)
        
        if epsilons==0:
            learning_rate_a = 0.0001
            
        if reward == 1:
            rewards_per_episode[i] = 1
            
            
    env.close()
    
    if is_training:
        f = open("FrozenLake/frozen_lake.pkl", "wb")
        pickle.dump(q, f)
        f.close()
    
    mean_reward = np.zeros(episodes)
    for t in range(episodes):
        mean_reward[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_reward)
    plt.savefig('FrozenLake/frozen_lake.png')
    
if __name__ == "__main__":
    run(15000, is_training=True, render=False)
    