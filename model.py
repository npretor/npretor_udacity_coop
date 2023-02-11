# Things I didn't remember: 
# .  where does the torch template go? it should inherit super. I suppose that would be the class doing the inheritance. 
# Fixed: input nn.Module, then super 

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    I'm creating these models from scratch 
    """
    def __init__(self, input_size, output_size, seed, settings):
        """
        
        """
        # This inherits from nn.Module. Enforces it as as template 
        super().__init__()
        self.fc1 = nn.Linear(input_size, settings['actor_network'][0]) 
        self.fc2 = nn.Linear(settings['actor_network'][1], output_size) 

    def forward(self, obs_size):
        x = F.relu(self.fc1(obs_size)) 
        x = F.relu(self.fc2(x)) 
        return x


class Critic(nn.Module):
    """
    I'm creating these models from scratch
    """
    def __init__(self, all_agent_obs_size, output_size, seed, settings):
        """
        
        """
        # This inherits from nn.Module. Enforces it as as template 
        super().__init__()

        self.fc1 = nn.Linear(all_agent_obs_size, settings['actor_network'][0]) 
        
        # TODO: size of fc2 input is wrong
        self.fc2 = nn.Linear(settings['actor_network'][1], output_size) 

    def forward(self, all_agent_obs, all_agent_actions):
        x = F.relu(self.fc1(all_agent_obs)) 

        # Need to concatenate the action in here to x
        x = torch.cat(all_agent_actions, x)

        x = F.relu(self.fc2(x)) 
        # Our agent's action. How does it know which one it is? Or should the action be just it's own action
        return x