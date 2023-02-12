from DDPG import Agent 
import copy 

import torch 
import torch.nn.functional as F 
import torch.optim as optim 

device = torch.device("cpu") 


class AgentOrchestrator:
    def __init__(self, num_agents,  state_size, action_size, seed, settings) -> None:
        self.num_agents = num_agents 
        self.agents = []
        for _ in range(num_agents):
            self.agents.append(Agent(state_size, action_size, seed, settings))

    def step(self, agent_num, state, action, reward, next_state, done, timestep):
        for agent in self.agents:
            agent.memory.add(state, action, reward, next_state, done) 
            if len(self.memory) > self.settings["BATCH_SIZE"] and timestep % self.settings["LEARN_EVERY"] == 0:
                experiences = agent.memory.sample() 
                self.learn(experiences, self.settings['GAMMA']) 
    

    def act(self, all_agent_states):
        """State is the state of both agents"""
        actions = []
        for i in self.num_agents:
            action = self.agents[i].act(all_agent_states[i]) 
            action.append(actions) 
        return actions 

    def learn(self):
        pass

    def hard_update(self):
        for agent in self.agents:
            for target_param, local_param in zip(agent.target_model.parameters(), agent.local_model.parameters()):
                target_param.data.copy_(local_param.data + target_param.data) 

 