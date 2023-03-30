import random, time, json
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from maddpg import AgentOrchestrator 
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
    average_scores = [] 
    max_reward = 0.0

    for ith_episode in range(1, num_episodes+1): 
        # Reset the environment and get the starting states
        env_info = env.reset(train_mode=True)[brain_name]     
        global_states = env_info.vector_observations 
        agents_scores = np.zeros(num_agents)
        episode_rewards = []               

        maddpg.noise_scalar = maddpg.noise_scalar - maddpg.noise_decay
        if maddpg.noise_scalar <= 0.0:
            maddpg.noise_scalar = 0.0   

        for timestep in range(max_timesteps):
            
            global_actions = maddpg.act(global_states, full_random=False)   
            #
            # print('actions: ',global_actions)

            env_info = env.step(global_actions)[brain_name]           
            global_next_states = env_info.vector_observations         
            global_rewards = env_info.rewards                         
            global_dones = env_info.local_done  

            maddpg.step(global_states, global_actions, global_rewards, global_next_states, global_dones) 

            agents_scores += global_rewards
            episode_rewards.append(global_rewards) 
            global_states = global_next_states
            
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

        wandb.log({
                "episode":ith_episode,
                "max_reward": max_reward,
                "moving_average_score": avg_score
            })        

        print("Episode: {}\t Avg score: {}\t Score: {}\t MaxR: {}\t Bffr: {}\t Noise: {}".format(
            ith_episode, 
            round(avg_score, 5), 
            round(episode_reward, 5), 
            max_reward,
            len(maddpg.memory),
            maddpg.noise_scalar)) 
        
        goal = 0.51
        if avg_score >= goal:
            print("success, saving model")
            for n, agent in enumerate(maddpg.agents):
                torch.save(agent.actor_local.state_dict(), "success_checkpoint_actor_{}.pth".format(n))   
                torch.save(agent.critic_local.state_dict(), "success_checkpoint_critic_{}.pth".format(n)) 
            goal += 0.1


training(settings['num_episodes'], settings['max_timesteps'])

env.close() 