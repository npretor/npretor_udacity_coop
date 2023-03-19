import random, time, json
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
from MADDPG import AgentOrchestrator 
import wandb
import torch 

with open("hyperparameters.json", 'r') as f:
    settings = json.load(f)

wandb.init(project="udacity_maddpg", config=settings) 

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64") 

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
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

        env_info = env.reset(train_mode=True)[brain_name]     
        global_states = env_info.vector_observations 
        agents_scores = np.zeros(num_agents)
        startTime = time.time()                
        currentTimesteps = 0                        

        for timestep in range(max_timesteps):
            #rand_actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            #global_actions = np.clip(rand_actions, -1, 1)                  # all actions between -1 and 1
            
            global_actions = maddpg.act(global_states)   

            env_info = env.step(global_actions)[brain_name]           # send all actions to tne environment 
            global_next_states = env_info.vector_observations         # get next state (for each agent) 
            global_rewards = env_info.rewards                         # get reward (for each agent) 
            global_dones = env_info.local_done                        # see if episode finished    

            # Flatten the states 
            global_states = global_states.reshape(-1) 
            global_actions = global_actions.reshape(-1) 
            global_rewards = np.asarray(global_rewards).reshape(-1) 
            global_next_states = global_next_states.reshape(-1) 
            global_dones = np.asarray(global_dones).reshape(-1) 
            
            maddpg.step(global_states, global_actions, global_rewards, global_next_states, global_dones, timestep) 

            agents_scores += env_info.rewards                         # update the score (for each agent)

            global_states = global_next_states.reshape(maddpg.num_agents, -1) 
            
            if np.any(global_dones): 
                break

        score = np.max(agents_scores) 
        scores.append(score) 
        
        scores_deque.append(score) 
        avg_score = np.mean(scores_deque)
        maddpg.current_avg_score = avg_score

        wandb.log({
                "episode":ith_episode,
                "score": score,
                "moving_average_score": avg_score
            })        

        print("Episode: {}\t Average score: {}\t Score: {}".format(ith_episode, avg_score, score)) 
        if avg_score >= 0.5:
            print("success, saving model")
            for n, agent in enumerate(maddpg.agents):
                torch.save(agent.actor_local.state_dict(), "success_checkpoint_actor_{}.pth".format(n))   
                torch.save(agent.critic_local.state_dict(), "success_checkpoint_critic_{}.pth".format(n)) 


training(settings['num_episodes'], settings['max_timesteps'])

env.close() 