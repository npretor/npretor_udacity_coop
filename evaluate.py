import random, time, json
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from maddpg import DemoAgents, AgentOrchestrator
import torch 

NUM_EPISODES = 10
MAX_TIMESTEPS = 1000

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


# maddpg = DemoAgents(
#     num_agents=num_agents,  
#     state_size=state_size, 
#     action_size=action_size, 
#     seed=10, 
#     checkpoints_folder='./', 
#     settings=settings
#     ) 

maddpg = AgentOrchestrator(2, state_size, action_size, 10, settings)
maddpg.load_checkpoint('./checkpoints/grateful-river-62_128x64')

scores_deque = deque(maxlen=100)
scores = []
average_scores = [] 
max_reward = 0.0

for ith_episode in range(NUM_EPISODES):
    env_info = env.reset(train_mode=False)[brain_name]     
    global_states = env_info.vector_observations 
    agents_scores = np.zeros(num_agents)
    episode_rewards = []        

    for _ in range(MAX_TIMESTEPS):
        global_actions = maddpg.act(global_states, full_random=False, add_noise=False)   
        global_actions = global_actions.reshape(2,2)
        env_info = env.step(global_actions)[brain_name]        

        print(global_actions)
           
        global_rewards = env_info.rewards                         
        global_dones = env_info.local_done  
        agents_scores += global_rewards
        episode_rewards.append(global_rewards) 
        
        if np.any(global_dones): 
            break


    episode_reward = np.max(np.sum(np.array(episode_rewards), axis=0))

    scores.append(episode_reward)
    scores_deque.append(episode_reward) 
    avg_score = np.mean(scores_deque)
    average_scores.append(avg_score)
    maddpg.current_avg_score = avg_score

    if episode_reward > max_reward:
        max_reward = episode_reward
    
        print("Episode: {}\t Avg score: {}\t Score: {}\t MaxR: {}\t Bffr: {}\t Noise: {}".format(
            ith_episode, 
            round(avg_score, 5), 
            round(episode_reward, 5), 
            max_reward,
            len(maddpg.memory),
            maddpg.noise_scalar)) 

env.close() 