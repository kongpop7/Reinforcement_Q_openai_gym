import numpy as np
import gym
import random
import time

env = gym.make("Blackjack-v1")
env.reset()

#Observation Space: Current sum of player (0-31), Face up card of the dealer (0-10), Have Ace (0-1)?
def adj_state(state):
    if state[2]:
        return (state[0],state[1],1)
    else:
        return (state[0],state[1],0)

epsilon = 1.0
episode = 5000
decay_rate = epsilon/episode
gamma = 0.8
learning = 0.5 
show_episode = 5

obs_size = [32,11,2]
num_actions = env.action_space.n
q_table = np.random.uniform(high = 1, low = -1, size = (obs_size + [env.action_space.n])) 

for i in range(episode):
    done = False
    state = env.reset()
    state = adj_state(state)
    if i>=(episode-show_episode):
        print('--------------------------------------')
    while done != True:   
        if i >= (episode - show_episode):
            print('Current_state: {}'.format(state))
            env.render()
            time.sleep(1)


        rd = np.random.random()

        if rd > epsilon:
            action = np.argmax(q_table[state]) 
        else:
            action = np.random.randint(0, env.action_space.n)

        if i>=(episode-show_episode):
            print('Player_action: {}'.format(action))
            if(action==0):
                print('Dealer score: {}'.format(sum(env.dealer)))
            
        state2, reward, done, info = env.step(action) 
        
        # if done and state2[0] >= 0.5:
        #     Q[state_adj[0], state_adj[1], action] = reward
        # else:
        # reward = reward_func(state2,state,reward)

        delta = learning*(reward + gamma*np.max(q_table[state]) - q_table[state][action])
        q_table[state][action] += delta
                                
        state = adj_state(state2)
    if i>=(episode-show_episode):
        print('Last player state: {}'.format(state))
        if reward == 1:
            print('WIN!')
        elif reward == 0:
            print('TIE!')
        else:
            print('LOSE!')
        print('--------------------------------------')
    
    #slowly decrease exploration factor 
    epsilon -= decay_rate
    
    if i%1000 == 0:
        print("Episodes: {}/{}     Epsilon:{}".format(i, episode,epsilon))
