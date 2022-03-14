#Number of rows and cols : 5x5, number of passenger's locations: 5, number of destination: 4
#Number of actions available: 6: South, North, East, West, Pickup, Dropoff
import gym
import numpy as np

#For discretizing the state
def discretize(state, bucket_size, max_size = 14):
    discrete_state = (state - env.observation_space.low)/bucket_size
    
    #clip to make sure it the state format never exceeds the bucket size 
    discrete_state = np.clip(discrete_state.astype(int), None, max_size)
    
    return tuple(discrete_state)

def reward_func(state, old_state, reward):
    # 0: cos1
    # 1: sin1
    # 2: cos2
    # 3: sin2 
    # 4: w1
    # 5: w2
    current_h = state[1]+state[1]*state[2]+state[3]*state[0]
    old_h = old_state[1]+old_state[1]*old_state[2]+old_state[3]*old_state[0]
    if current_h > old_h:
        reward += 2
    
    # Give reward if the stick is linear
    if (1-abs(state[2])<0.2) and (abs(state[3])<0.2):
        reward += 0.5
    
    # Give reward if the velocity is increasing
    if state[4]>old_state[4]:
        reward +=0.5
    
    return reward


env = gym.make("Acrobot-v1")
env.reset()

high = env.observation_space.high
low = env.observation_space.low

#Discretize the observation space
buckets = 15 
discrete_obs_size = [buckets] * len(env.observation_space.low)  
bucket_size = (env.observation_space.high-env.observation_space.low)/discrete_obs_size

#create q table to match all possible states with actions
q_table = np.random.uniform(high = 1, low = -1, size = (discrete_obs_size + [env.action_space.n])) 

episode = 3000
gamma = 0.99
learning = 1e-3

#exploration vs exploitation factor
eps = 1
eps_decay_rate = eps/episode

num_actions = env.action_space.n

for i in range(episode):
    print(i)
    done = False
    state = env.reset()
    state = discretize(state,bucket_size)

    while done != True:   
        if i >= (episode - 10):
            env.render()

        rd = np.random.random()

        if rd > eps:
            action = np.argmax(q_table[state]) 
        else:
            action = np.random.randint(0, env.action_space.n)
            
        state2, reward, done, info = env.step(action) 
        
        # if done and state2[0] >= 0.5:
        #     Q[state_adj[0], state_adj[1], action] = reward
        # else:
        reward = reward_func(state2,state,reward)

        delta = learning*(reward + gamma*np.max(q_table[state]) - q_table[state][action])
        q_table[state][action] += delta
                                
        state = discretize(state2,bucket_size)
    
    #slowly decrease exploration factor 
    eps -= eps_decay_rate
    
    if i%50 == 0:
        print("Episodes: {}/{}     Epsilon:{}".format(i, episode,eps))
