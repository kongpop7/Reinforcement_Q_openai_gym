# environment

import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple

import glob
import io
import base64
import os
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor

display = Display(visible=0, size=(1400, 900))
display.start()

if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    !bash ../xvfb start
    %env DISPLAY=:1
      
def show_videos():
  mp4list = glob.glob('video/*.mp4')
  mp4list.sort()
  for mp4 in mp4list:
    print(f"\nSHOWING VIDEO {mp4}")
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))

def wrap_env(env, video_callable=None):
  env = Monitor(env, './video', force=True, video_callable=video_callable)
  return env   

env = gym.make('Acrobot-v1')
env.seed(0)
env = wrap_env(env, video_callable=lambda episode_id: True)

for num_episode in range(10): 
    state = env.reset()
    score = 0
    done = False
    # Go on until the pole falls off or the score reach -500
    while not done and score > -500:
      # Choose a random action
      action = random.choice([0, 1, 2])
      next_state, reward, done, info = env.step(action)
      # Visually render the environment 
      env.render()
      # Update the final score (-1 for each step)
      score += reward 
      state = next_state
      # Check if the episode ended (the pole fell down)
    print(f"EPISODE {num_episode + 1} - FINAL SCORE: {score}") 