import random
from collections import namedtuple, deque
import copy 
import torch 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np
from ddpg_model import Actor, Critic

device = torch.device("cpu") 

"""
Tasks 
1. Write and verify the agent is adding experiences 
"""


class DemoAgent():
    def __init__(self, state_size, action_size, seed, actor_path, critic_path, settings) -> None:
        self.actor= Actor(state_size, action_size, random.random(), settings['actor_network_shape']).to(device)
        self.critic = Critic(state_size, action_size, random.random(), settings['critic_network_shape']).to(device)
        self.actor.load_state_dict(torch.load(actor_path))   
        self.critic.load_state_dict(torch.load(critic_path)) 

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) 
        self.actor.eval()
        self.critic.eval() 
        with torch.no_grad():
            action_values = self.actor(state) 
        return action_values.cpu().data.numpy()         


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
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.settings["LR_CRITIC"]) #, weight_decay=self.settings["WEIGHT_DECAY"])

        # Noise process
        #self.noise = OUNoise(action_size * num_agents, random_seed)
        # self.noise = RandNoise(action_size*num_agents, 10)
        self.noise = RandNoise(action_size, 10)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, self.settings["BUFFER_SIZE"], self.settings["BATCH_SIZE"], random_seed)

        # Target and local should be the same for actor and critic
        for target_param, local_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(local_param.data)  
            
        for target_param, local_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data) 
    
    def act(self, state, noise_decay_rate, add_noise=True):
        """
        states(list): 
            list of states for each agent 
        Returns: 
            actions for given state as per current policy 
        """
        # Convert states from numpy to tensor 
        state = torch.from_numpy(state).float().to(device) 
        
        # Set network to eval mode (as opposed to training mode)
        self.actor_local.eval() 

        # Get a state from actor_local and add it to the list of states for each actor  
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() 

        self.actor_local.train()
        if add_noise:
            #self.noise_decay_rate = 1 - (self.current_avg_score/self.goal_avg_score)
            action += self.noise.sample().reshape((-1)) * noise_decay_rate   
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agent_index, gamma):
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
        if agent_index == 0:
            other_index = 1
        else:
            other_index = 0

        batch_size = self.settings['BATCH_SIZE'] 
        num_global_agents = 2

        global_states, global_actions, global_rewards, global_next_states, global_dones = experiences

        # [batch_size, 2, ...]
        global_states =         global_states.reshape(batch_size, num_global_agents, -1)  
        global_actions =        global_actions.reshape(batch_size, num_global_agents, -1) 
        global_rewards =        global_rewards.reshape(batch_size, num_global_agents, -1) 
        global_next_states =    global_next_states.reshape(batch_size, num_global_agents, -1) 
        global_dones =          global_dones.reshape(batch_size, num_global_agents, -1)  

        global_next_actions = torch.hstack((self.actor_target(global_next_states[:,agent_index,:]), self.actor_target(global_next_states[:,other_index,:])  )) 
        global_next_actions_pred = torch.hstack((self.actor_target(global_states[:,agent_index,:]), self.actor_target(global_states[:,other_index,:])  )) 

        local_rewards = global_rewards[:,agent_index,:] # .view(-1, 1) 
        local_dones   = global_dones[:,agent_index,:]  # .view(-1, 1)  

        #import ipdb; ipdb.set_trace() 


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # THIS TILL FAIL ON MORE THAN 2 AGENTS


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
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)     
        self.critic_optimizer.step() 

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        

        # critic_local( [256, 48], [256, 4]) 
        actor_loss = -self.critic_local(global_states.reshape(batch_size, -1), global_next_actions_pred.reshape(batch_size, -1)).mean() 
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
        θ_target = τ*θ_local + (1 - τ)*θ_target

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


class AgentOrchestrator:
    def __init__(self, num_agents,  state_size, action_size, seed, settings):
        self.settings = settings 
        self.num_agents = num_agents 
        self.current_avg_score = 0.001

        #  - - - Set the noise falloff - - - -  # 
        #  - - - - Linear fallof  - - - - - - - #
        # Let's set this in proximity to the goal. I could set a linear value, but that's not very helpful. 
        # Rather, if I set the noise multiplier amount proportional the proximity to a stated goal, it should auto-falloff as I approach
        # This method only works because I know I want to approach a score of 30
        self.goal_avg_score = .51
        if self.current_avg_score > self.goal_avg_score:
            self.current_avg_score = self.goal_avg_score
        else:
            self.current_avg_score = self.current_avg_score 
        self.noise_decay_rate = abs(1 - (self.current_avg_score/self.goal_avg_score))  

        self.agents = [] 
        for id in range(num_agents): 
            self.agents.append(Agent(state_size, action_size, seed, settings, 2, id)) 
        
        self.memory = ReplayBuffer(action_size, self.settings["BUFFER_SIZE"], self.settings["BATCH_SIZE"], seed=55) 


    def learn(self, experiences, gamma):
        for agent_index, agent in enumerate(self.agents):
            agent.learn(experiences, agent_index, gamma) 



    def step(self, global_states, global_actions, global_rewards, global_next_states, global_dones, timestep):
        """ 
        Add global experiences to memory  
        """
        self.memory.add(global_states, global_actions, global_rewards, global_next_states, global_dones) 

        if len(self.memory) > self.settings["BATCH_SIZE"] and timestep % self.settings["LEARN_EVERY"] == 0:
            N_RETRAININGS = 4
            for _ in range(N_RETRAININGS):
                experiences = self.memory.sample() 
                for agent_index, agent in enumerate(self.agents):
                    agent.learn(experiences, agent_index, self.settings['GAMMA']) 
                #self.learn(experiences, self.settings['GAMMA']) 
    

    def act(self, all_agent_states):
        """State is the state of both agents"""
        actions = np.empty((self.num_agents, 2))    # This should be clipped 
        for i, agent in enumerate(self.agents):
            #self.noise_decay_rate = abs(1 - (self.current_avg_score/self.goal_avg_score) )
            self.noise_decay_rate = 1.0
            actions[i] = agent.act(all_agent_states[i], self.noise_decay_rate) 

        return actions  # np.clip(action, -1, 1)
