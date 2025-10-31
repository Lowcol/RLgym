import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    
    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:
        f = open("mountain_car.pkl", "rb")
        q = pickle.load(f)
        f.close()
        
        
    learning_rate_a = 0.9 #alpha or learning rate
    discount_factor_g = 0.9 #gamma or discount factor
    
    epsilons = 1 # 1 = 100% random action
    epsilon_decay_rate = 2/episodes 
    rng = np.random.default_rng() # random number generator
    
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space) #to figure out which segment the position belongs to
        state_v = np.digitize(state[1], vel_space) 
        
        terminated = False
        
        rewards = 0
        
        while(not terminated and rewards > -1000):
            
            if is_training and rng.random() < epsilons:
                #choose random action (0=drive left, 1=do nothing, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])
            
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            
            if is_training:
                #update q value
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )
            
            state = new_state  
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward
        
        epsilons = max(0, epsilons - epsilon_decay_rate)
        
        rewards_per_episode[i] = rewards
        
    env.close()
    
    if is_training:
        f = open("mountain_car.pkl", "wb")
        pickle.dump(q, f)
        f.close()
    
    mean_reward = np.zeros(episodes)
    for t in range(episodes):
        mean_reward[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_reward)
    plt.savefig('mountain_car.png')
    
if __name__ == "__main__":
    run(4000, is_training=False, render=True)
    