import copy 
from DDPG import Agent
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
        self.agents = [] 
        for _ in range(num_agents): 
            self.agents.append(Agent(state_size, action_size, seed, settings, 1)) 


    def learn(self, experiences, gamma):
        """
        Experiences structure: 
        experiences = [ [s, a, ś, r, done], [s, a, ś, r, done] ] 
        """ 

        for n in range(self.num_agents): 
            # TODO: Not sure 
            # 1. Need to update the agent learn function to accept full experiences and parse them out 
            # 2. 
            self.agents[n].learn(experiences, gamma) 
        
        self.hard_update() 


    def step(self, states, actions, rewards, next_states, dones, timestep):
        """
         
        Train on: 
            State of all agents 
            Actions of   
        """

        # for agent in self.agents:
        for i in range(self.num_agents):
            self.agents[i].memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i]) 
            
            if len(self.agents[i].memory) > self.settings["BATCH_SIZE"] and timestep % self.settings["LEARN_EVERY"] == 0:
                experiences = self.agents[i].memory.sample() 
                self.learn(experiences, self.settings['GAMMA']) 
    

    def act(self, all_agent_states):
        """State is the state of both agents"""
        actions = np.empty((self.num_agents, 2))
        for i in range(self.num_agents):
            actions[i] = self.agents[i].act(all_agent_states[i]) 
        return actions 



    def hard_update(self):
        #for agent in self.agents:
        for target_param, local_param in zip(self.agents[0].target_model.parameters(), self.agents[1].local_model.parameters()):
            target_param.data.copy_(local_param.data + target_param.data) 