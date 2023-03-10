import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cpu") 


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, settings, num_agents, agent_id):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.settings = settings
        self.num_agents = num_agents 
        self.agent_id = agent_id

        # Let's set this in proximity to the goal. I could set a linear value, but that's not very helpful. 
        # Rather, if I set the noise multiplier amount proportional the proximity to a stated goal, it should auto-falloff as I approach
        # This method only works because I know I want to approach a score of 30
        self.current_avg_score = 0.001 
        self.goal_avg_score = 30;
        
        # This should be zero or close to the reward or above, and approaching one as the current_avg approaches zero 
        #self.noise_decay_rate = 1 - (self.goal_avg_score - self.current_avg_score)      
        self.noise_decay_rate  = 1
        self.actor_loss = 0.0 
        self.critic_loss = 0.0 

        # Actor Network (w/ Target Network)
        self.actor_local  = Actor(state_size, action_size, random_seed, settings["actor_network_shape"]).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, settings["actor_network_shape"]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.settings["LR_ACTOR"])

        # Critic Network (w/ Target Network)
        self.critic_local  = Critic(state_size, action_size, random_seed, settings["critic_network_shape"]).to(device) 
        self.critic_target = Critic(state_size, action_size, random_seed, settings["critic_network_shape"]).to(device) 
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.settings["LR_CRITIC"], weight_decay=self.settings["WEIGHT_DECAY"])

        # Noise process
        self.noise = OUNoise(action_size * num_agents, random_seed)
        #self.noise = RandNoise(action_size*num_agents, 10)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.settings["BUFFER_SIZE"], self.settings["BATCH_SIZE"], random_seed)

        # Copy networks
        #  
        for target_param, local_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(local_param.data + target_param.data) 
            
        for target_param, local_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data + target_param.data)             

    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done) 

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.settings["BATCH_SIZE"] and timestep % self.settings["LEARN_EVERY"] == 0:
            experiences = self.memory.sample()
            self.learn(experiences, self.settings["GAMMA"]) 

    def act(self, state, add_noise=True):
        """
        states(list): 
            list of states for each agent 
        Returns: 
            actions for given state as per current policy 
        """
        # Convert states from numpy to tensor 
        #import ipdb; ipdb.set_trace()

        state = torch.from_numpy(state).float().to(device) 
        
        # Set network to eval mode (as opposed to training mode)
        self.actor_local.eval() 


        # Get a state from actor_local and add it to the list of states for each actor  
        with torch.no_grad():
            #actions[i] = self.actor_local(states[i]).cpu().data.numpy() 
            action = self.actor_local(state).cpu().data.numpy() 

        action_size = 2

        self.actor_local.train()
        if add_noise:
            #action += self.noise.sample().reshape((-1, action_size)) * self.noise_decay_rate
            action += self.noise.sample().reshape((-1)) # * self.noise_decay_rate   
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agent_index, gamma):
        """
        This function differs from the single agent DDPG in that the critic evaluates different material 
        
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ?? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        batch_size = self.settings['BATCH_SIZE'] 

        #global_states, actions, rewards, next_states, dones = experiences
        global_states, global_actions, global_rewards, global_next_states, global_dones = experiences
        num_all_agents = 2

        global_states =         global_states.reshape(batch_size, num_all_agents, -1)  
        global_actions =        global_actions.reshape(batch_size, num_all_agents, -1) 
        global_rewards =        global_rewards.reshape(batch_size, num_all_agents, -1) 
        global_next_states =    global_next_states.reshape(batch_size, num_all_agents, -1) 
        global_dones =          global_dones.reshape(batch_size, num_all_agents, -1)  


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = None
        # THIS TILL FAIL ON MORE THAN 2 AGENTS
        for i in range(num_all_agents - 1):
            # actions_next.append(self.actor_target(global_next_states[:,i,:]))
            actions_next = torch.hstack((self.actor_target(global_next_states[:,i,:]), self.actor_target(global_next_states[:,i+1,:])  ))
        
        # self.actor_target(global_next_states[:,i,:]) 
        
        # Each action is received as torch.Size([64,2])
        # |
        # V
        # We want a torch.shape of [64,4]  
        #actions_next = np.array(actions_next, dtype=float) 
        #import ipdb; ipdb.set_trace() 

        Q_targets_next = self.critic_target(global_next_states.reshape(batch_size, -1), actions_next) 

        # Compute Q targets for current states (y_i) 
        #Q_targets = global_rewards[agent_index] + (gamma * Q_targets_next[agent_index] * (1 - global_dones[agent_index])) 
        Q_targets = global_rewards[:,agent_index,:] + (gamma * Q_targets_next * (1-global_dones[:,agent_index,:]))

        # Compute critic loss
        Q_expected = self.critic_local(global_states.reshape(batch_size, -1), global_actions.reshape(batch_size, -1)) 
        critic_loss = F.mse_loss(Q_expected, Q_targets) 
        self.critic_loss = critic_loss

        # Minimize the loss
        self.critic_optimizer.zero_grad() 
        critic_loss.backward() 

        # Taken from: https://github.com/adaptationio/DDPG-Continuous-Control/blob/master/agent.py
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)     
        self.critic_optimizer.step() 

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        for i in range(num_all_agents - 1):
            # actions_next.append(self.actor_target(global_next_states[:,i,:]))
            actions_next = torch.hstack((self.actor_target(global_states[:,i,:]), self.actor_target(global_states[:,i+1,:])  ))

        # actions = []
        # for i in range(num_all_agents):
        #     actions.append(self.actor_target(global_states[:,i,:]))
        # actions = np.array(actions, dtype=np.float) 

        # actions_pred = self.actor_local(global_states[agent_index])
        actor_loss = -self.critic_local(global_states.reshape(batch_size, -1), actions_next.reshape(batch_size, -1)).mean()
        self.actor_loss = actor_loss 

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.settings["TAU"]) 
        self.soft_update(self.actor_local, self.actor_target, self.settings["TAU"]) 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2): 
        """Initialize parameters and noise process.""" 
        self.mu = mu * np.ones(size) 
        self.theta = theta 
        self.sigma = sigma 
        self.seed = random.seed(seed) 
        self.reset() 

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class RandNoise:
    def __init__(self, size, seed) -> None:
        self.size = size
        self.shape = (2,2)
        pass


    def reset(self):
        pass 

    def sample(self):
        rand_actions = np.random.randn(self.size) 
        rand_actions = np.clip(rand_actions, -1, 1)    
        return rand_actions 


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

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
