import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make("Taxi-v3", render_mode="human" if render else None)

    
    if is_training:
        q1 = np.zeros((env.observation_space.n, env.action_space.n))
        q2 = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("taxi/taxi.pkl", "rb")
        q1 = pickle.load(f)
        q2 = pickle.load(f)
        f.close()
        
        
    learning_rate_a = 0.1 #alpha or learning rate
    discount_factor_g = 0.95 #gamma or discount factor
    epsilons = 0.1 # 1 = 100% random action
    epsilon_decay_rate = 0.995 # decay per episode
    rng = np.random.default_rng() # random number generator
    
    rewards_per_episode = np.zeros(episodes)
    
    
    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}")
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0


        while(not terminated and not truncated):
            if is_training and rng.random() < epsilons:
                #choose random action (0=move left, 1=move down, 2=move right, 3=move up)
                action = env.action_space.sample()
            else:
                action = np.argmax(q1[state, :] + q2[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if is_training:
                if rng.random() < 0.5:
                    # Update q1
                    q1[state, action] = q1[state, action] + learning_rate_a * (
                        reward + discount_factor_g * q2[new_state, np.argmax(q1[new_state, :])] - q1[state, action]
                    )
                else:
                    # Update q2
                    q2[state, action] = q2[state, action] + learning_rate_a * (
                        reward + discount_factor_g * q1[new_state, np.argmax(q2[new_state, :])] - q2[state, action]
                    )


            state = new_state
            episode_reward += reward
            
        
        epsilons = max(0.01, epsilons * epsilon_decay_rate)
        
        rewards_per_episode[i] = episode_reward
            
            
    env.close()
    
    if is_training:
        f = open("taxi/taxi.pkl", "wb")
        pickle.dump(q1, f)
        pickle.dump(q2, f)  
        f.close()
    
    mean_reward = np.zeros(episodes)
    for t in range(episodes):
        mean_reward[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_reward)
    plt.title('Double Q-Learning - Taxi')
    plt.savefig('taxi/taxi.png')
    
if __name__ == "__main__":
    run(5000, is_training=True, render=False)
    