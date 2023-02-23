import random, time, json
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
from MADDPG import AgentOrchestrator 


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

maddpg = AgentOrchestrator(num_agents=num_agents,  state_size=state_size, action_size=action_size, seed=10, settings=settings)

def training(num_episodes, max_timesteps=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    avg_scores = []
    max_score = -np.Inf     

    for ith_episode in range(1, num_episodes+1): 
        # Reset the environment and get the starting states

        env_info = env.reset(train_mode=False)[brain_name]     
        global_states = env_info.vector_observations 

        startTime = time.time()                
        agents_scores = np.zeros(num_agents)  
        currentTimesteps = 0                        

        for timestep in range(max_timesteps):

            rand_actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            rand_actions = np.clip(rand_actions, -1, 1)                  # all actions between -1 and 1
            #import pdb; pdb.set_trace()
            
            global_actions = maddpg.act(global_states) 
            #print('Action:',actions)

            env_info = env.step(global_actions)[brain_name]           # send all actions to tne environment
            global_next_states = env_info.vector_observations         # get next state (for each agent)
            global_rewards = env_info.rewards                         # get reward (for each agent)
            global_dones = env_info.local_done                        # see if episode finished
            
            maddpg.step(global_states, global_actions, global_rewards, global_next_states, global_dones, timestep) 


            
            scores += env_info.rewards                         # update the score (for each agent)
            global_states = global_next_states                               # roll over states to next time step
            if np.any(global_dones):                                  # exit loop if episode finished
                break

        print('Score (max over agents) from episode {}: {}'.format(ith_episode, np.max(scores))) 
        print('Average score: ',np.mean(scores))

training(settings['num_episodes'], settings['max_timesteps'])

env.close()