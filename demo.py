import random, time, json
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from maddpg import DemoAgents
import torch 

# Startup the Unity environment 
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64") 
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]

num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]


with open("hyperparameters.json", 'r') as f:
    settings = json.load(f)

agents = DemoAgents(num_agents, state_size, action_size, 10, './checkpoints/grateful-river-62_128x64/', settings)

for i in range(10):
    env_info = env.reset(train_mode=False)[brain_name]     
    states = env_info.vector_observations 
    for _ in range(1000):
        actions = agents.act(states).reshape(1, 4)
        print(actions)

        env_info = env.step(actions)[brain_name] 

        dones = env_info.local_done

        if np.any(dones): 
            break