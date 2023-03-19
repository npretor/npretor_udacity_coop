import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, network_shape):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed) 
        self.fc1 = nn.Linear(state_size, network_shape[0])

        self.bn1 = nn.BatchNorm1d(network_shape[0]) 
        self.fc2 = nn.Linear(network_shape[0], network_shape[1]) 
        self.fc3 = nn.Linear(network_shape[1], action_size) 
        self.reset_parameters() 

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions.""" 
        #import ipdb; ipdb.set_trace()  
        if state.ndim == 1:
            state = torch.unsqueeze(state, 0)         

        x = F.relu(self.fc1(state)) 

        x = self.bn1(x) 
        x = F.relu(self.fc2(x)) 
        return torch.tanh(self.fc3(x)) 


class Critic(nn.Module):
    """Critic (Value) Model."""


    def __init__(self, state_size, action_size, seed, network_shape):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): One agent's state size
            action_size (int): One agent's action size
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        #self.num_agents = num_agents 
        self.seed =     torch.manual_seed(seed) 
        self.fcs1 =     nn.Linear(state_size*2, network_shape[0])
        self.bn1 =      nn.BatchNorm1d(network_shape[0]) 
        self.fc2 =      nn.Linear(network_shape[0] + action_size*2, network_shape[1]) 
        #self.fc3 =     nn.Linear(network_shape[1], network_shape[2])
        self.fc4 =      nn.Linear(network_shape[1], 1)
        self.reset_parameters()


    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # State = Tensor[1024, 24]
        # action = Tensor[1024, 2] 
        
        xs = F.relu(self.fcs1(states)) 
        #if xs.ndim == 1:
        #    xs = torch.unsqueeze(xs, 0)  
        xs = self.bn1(xs)
        x = torch.cat((xs, actions), dim=1)    #x = torch.vstack((xs, action))
        x = F.relu(self.fc2(x))
        return self.fc4(x)
