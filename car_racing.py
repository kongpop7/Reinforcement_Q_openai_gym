#Number of rows and cols : 5x5, number of passenger's locations: 5, number of destination: 4
#Number of actions available: 6: South, North, East, West, Pickup, Dropoff
import gym
import numpy as np

env = gym.make("Acrobot-v1")
env.reset()

high = env.observation_space.high
low = env.observation_space.low

obs_size = list((high-low).astype(int)*[10,10,10,1,1,1]+1)
print(obs_size)

epsilon = 1.0
episode = 500
decay_rate = 0.99
gamma = 0.8 
learning = 0.00001

num_actions = env.action_space.n

obs_size.append(num_actions)
Q = np.random.uniform(low = -1, high = 1,size = tuple(obs_size))
for i in range(episode):

    done = False
    state = env.reset()
    state = (state-low).astype(int)*[10, 10, 10, 1, 1, 1]

    while done != True:   
        if i >= (episode - 1):
            env.render()

        rd = np.random.random()

        if rd > epsilon:
            action = np.argmax(Q[state[0],state[1],state[2],state[3],state[4],state[5]]) 
        else:
            action = np.random.randint(0, env.action_space.n)
            
        state2, reward, done, info = env.step(action) 
        
        # if done and state2[0] >= 0.5:
        #     Q[state_adj[0], state_adj[1], action] = reward
        # else:

        delta = learning*(reward + gamma*np.max(Q[state[0],state[1],state[2],state[3],state[4],state[5]]) - Q[state[0],state[1],state[2],state[3],state[4],state[5],action])
        Q[state[0],state[1],state[2],state[3],state[4],state[5],action] += delta
                                
        state = (state2[0]-low).astype(int)*[10,10,10,1,1,1]
   
    print(i)
    epsilon -= decay_rate
