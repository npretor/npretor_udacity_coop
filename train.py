import random, time, json
from unityagents import UnityEnvironment
from collections import deque
import numpy as np


with open("hyperparameters.json", 'r') as f:
    settings = json.load(f)

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def training(num_episodes, max_timesteps=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    avg_scores = []
    max_score = -np.Inf     

    for ith_episode in range(1, num_episodes+1): 
        # Reset the environment and get the starting states

        startTime = time.time()                
        env_info = env.reset(train_mode=False)[brain_name]     
        agents_scores = np.zeros(num_agents)  
        states = env_info.vector_observations 
        currentTimesteps = 0                        

        for timestep in range(max_timesteps):
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break

        print('Score (max over agents) from episode {}: {}'.format(ith_episode, np.max(scores))) 


env.close()