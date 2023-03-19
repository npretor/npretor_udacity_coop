import copy 
from DDPG import Agent, ReplayBuffer
import torch 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np

device = torch.device("cpu") 

"""
Tasks 
1. Write and verify the agent is adding experiences 
"""


class AgentOrchestrator:
    def __init__(self, num_agents,  state_size, action_size, seed, settings):
        self.settings = settings 
        self.num_agents = num_agents 
        self.current_avg_score = 0
        self.agents = [] 
        for id in range(num_agents): 
            self.agents.append(Agent(state_size, action_size, seed, settings, 1, id)) 
        
        self.memory = ReplayBuffer(action_size, self.settings["BUFFER_SIZE"], self.settings["BATCH_SIZE"], 55)


    def learn(self, experiences, gamma):
        """
        This function differs from the single agent DDPG in that the critic evaluates different material 
        
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        batch_size = self.settings['BATCH_SIZE'] 

        global_states, global_actions, global_rewards, global_next_states, global_dones = experiences

        # [batch_size, 2, ...]
        global_states =         global_states.reshape(batch_size, self.num_agents, -1)  
        global_actions =        global_actions.reshape(batch_size, self.num_agents, -1) 
        global_rewards =        global_rewards.reshape(batch_size, self.num_agents, -1) 
        global_next_states =    global_next_states.reshape(batch_size, self.num_agents, -1) 
        global_dones =          global_dones.reshape(batch_size, self.num_agents, -1)  

        local_rewards = global_rewards[:,agent_index,:] # .view(-1, 1) 
        local_dones   = global_dones[:,agent_index,:]  # .view(-1, 1)   

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        global_next_actions = torch.hstack((self.actor_target(global_next_states[:,0,:]), self.actor_target(global_next_states[:,1,:])  )) 
    

        with torch.no_grad():
            Q_targets_next = self.critic_target(global_next_states.reshape(batch_size, -1), global_next_actions)  
            Q_targets = local_rewards + (gamma * Q_targets_next * (1-local_dones))

        # Compute critic loss
        Q_expected = self.critic_local(global_states.reshape(batch_size, -1), global_actions.reshape(batch_size, -1)) 
        critic_loss = F.mse_loss(Q_expected, Q_targets) 
        self.critic_loss = critic_loss

        # Minimize the loss
        self.critic_optimizer.zero_grad() 
        critic_loss.backward() 

        # Taken from: https://github.com/adaptationio/DDPG-Continuous-Control/blob/master/agent.py
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)     
        self.critic_optimizer.step() 

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        global_next_actions_pred = torch.hstack((self.actor_target(global_states[:,0,:]), self.actor_target(global_states[:,1,:])  )) 

        # critic_local( [256, 48], [256, 4]) 
        actor_loss = -self.critic_local(global_states.reshape(batch_size, -1), global_next_actions_pred.reshape(batch_size, -1)).mean() 
        self.actor_loss = actor_loss 

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.settings["TAU"]) 
        self.soft_update(self.actor_local, self.actor_target, self.settings["TAU"]) pass



    def step(self, global_states, global_actions, global_rewards, global_next_states, global_dones, timestep):
        """
        Critic learns from global states, actor learns from local info (a, s', r, done)
        Memory should look like: [
        [[agent1_experiences0], [agent2_experiences0]],
        [[agent1_experiences1], [agent2_experiences1]],
        ]
        Add the full experience to the replay buffer, and have the agents sort it out 
        """
        #experience = global_states, global_actions, global_rewards, global_next_states, global_dones
        self.memory.add(global_states, global_actions, global_rewards, global_next_states, global_dones) 


        if len(self.memory) > self.settings["BATCH_SIZE"] and timestep % self.settings["LEARN_EVERY"] == 0:
            experiences = self.memory.sample() 
            self.learn(experiences, self.settings['GAMMA']) 
    

    def act(self, all_agent_states):
        """State is the state of both agents"""
        actions = np.empty((self.num_agents, 2))
        for i in range(self.num_agents):
            actions[i] = self.agents[i].act(all_agent_states[i]) 
        return actions 
