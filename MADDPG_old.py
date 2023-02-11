import random 
from collections import namedtuple, deque
import torch 
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np

from maddpg_model import Actor, Critic

"""
A ddpg agent should have 

* Networks
    Actor
    Critic 

* Noise generator 
* Memory replay buffer 

* Setup
    Define actor_target 
    Define actor_actual 
    Define actor_actual_optimizer 

    Define critic_target 
    Define critic_actual 
    Define critic_actual_optimizer 

    Define noise 

    Define replay buffer 


* Soft copy function
    Copy parameters from target to actual for critic and actor 

* Step function
    Add state, action, next_state, reward, done to memory 
    Learn if samples have filled up the batch 


* Act function 
    action = actor_local.forward(state) 
    noise.sample() 
    action * noise * noise_decay 

* Learn function 
    >memory batch 
    actor_local.train() 


""" 

device = torch.device("cpu") 


class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed, settings) -> None:
        self.state_size = state_size
        self.action_size = action_size 
        self.settings = settings 
        self.num_agents = num_agents 

        self.noise_decay_rate = 1 - (self.goal_avg_score - self.current_avg_score)

        self.actor_local  = Actor(num_agents*state_size, action_size, random_seed, settings['actor_network_shape']).to(device) 
        self.actor_target = Actor(num_agents*state_size, action_size, random_seed, settings['actor_network_shape']).to(device) 
        self.critic_local  = Critic(state_size, action_size, random_seed, settings['critic_network_shape']).to(device)  
        self.critic_target = Critic(state_size, action_size, random_seed, settings['critic_network_shape']).to(device) 

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.settings['LR_CRITIC'], weight_decay=self.settings['WEIGHT_DECAY'])
        self.actor_optimizer  = optim.Adam(self.actor_local.parameters(),  lr=self.settings['LR_ACTOR'], weight_decay=self.settings['WEIGHT_DECAY']) 

        self.memory = ReplayBuffer(100) 
        self.settings = settings 
    
    def learn(self, experiences):
        """
        Get the experiences 
        Run states through the actor_target network and get the inferred next actions 
        Calculate the loss of the actor network 
        Get the predicted reward from the local_critic network 

        # Actor    f(state)          =>  a*  
        # Critic   f(state, action)  =>  Q  
        """
        
        states, actions, rewards, next_states, dones = experiences 
        gamma = self.settings['gamma']
        # Create two actor-critic combinations. 
        # A base/local version 
        # An exploratory/target version 
        # We train and update the exploratory version, and use it to 
        #   slowly update the base version to reduce noise

        # Train the critic 
        Q_local  = self.critic_local(states, actions) 
        Q_target_next = self.critic_target(next_states, self.actor_target(next_states)) 
        Q_target = rewards + (gamma * Q_target_next * (1 - dones)) 

        # Multiply the squared weights of each model to calculate the loss 
        critic_loss = F.mse_loss(Q_target, Q_local) 

        # Train the actor 
        actor_loss = -self.critic_local(states, self.actor_local(states)).mean() 

        self.critic_optimizer.zero_grad() 

        self.critic_optimizer.step() 
        self.actor_optimizer.step() 


        # TODO: soft update as the last thing. From -> to
        self.soft_update(self.target_actor, self.actual_actor)
        self.soft_update(self.target_actor, self.actual_actor) 
        


    def step(self, states, actions, rewards, next_states, dones, timestep):
        """
        Add samples to memory 
        Train on random samples from memory 
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done) 

        if len(self.memory) > self.settings['BATCH_SIZE'] and timestep % self.settings['LEARN_EVERY'] == 0:
            experiences = self.memory.sample()
            self.learn(experiences, self.settings['GAMMA']) 

    def act(self, state, add_noise=True):
        """
        Several explore options: 
            Threshold: Stop adding noise at a certain distance from the goal 
            Linear:  Falloff of noise based on distance from goal 
            Exponential: Fallof of noise based on distance from goal 

        Explore-exploit ratio: 

        Two kinds of noise: 
        - Choose a random action with a given ratio
        - Apply noise to the action, a lot at first and falling off as the goal approaches
        """

        state = torch.from_numpy(state).float().to(device) 
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() 
        
        action = action * self.noise.sample().reshape(-1, self.num_agents) * self.noise_decay_rate 

        return np.clip(action, -1, 1)


class ReplayBuffer:
    """Copied this from the last project, not rewriting this"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add batch of new experiences to memory."""
        # for i in range(len(states)):
        #     print("adding experience batch: {}".format(i))
        #     print(states[i])
        #     e = self.experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
        #     self.memory.append(e)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class NoiseGen:
    def __init__(self, sample_size) -> None:
        """
        Turns out we don't need OU noise. Let's keep it simple 
        """
        self.sample_size = sample_size

    def sample(self):
        """
        Random values in range between 0 and 1 of size sample_size 
        """
        return self.np.random.rand(self.sample_size)