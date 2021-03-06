import gym
import numpy as np

epsilon = 0.3 
gamma = 0.8 
learning = 0.2 

env = gym.make('MountainCar-v0')
env.reset()
env.step(0)

num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

Q = np.random.uniform(low = -1, high = 1,size = (num_states[0], num_states[1],env.action_space.n))
for i in range(5000):
    done = False
    state = env.reset()

    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)

    while done != True:   
        if i >= (5000 - 20):
            env.render()
        if i > 4000: 
            epsilon = 0.05 
        if np.random.random() < 1 - epsilon:
            action = np.argmax(Q[state_adj[0], state_adj[1]]) 
        else:
            action = np.random.randint(0, env.action_space.n)
            
        state2, reward, done, info = env.step(action) 
      
        state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
        state2_adj = np.round(state2_adj, 0).astype(int)
        
        if done and state2[0] >= 0.5:
            Q[state_adj[0], state_adj[1], action] = reward
        else:
            delta = learning*(reward + gamma*np.max(Q[state2_adj[0],state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])
            Q[state_adj[0], state_adj[1],action] += delta
                                
        state_adj = state2_adj