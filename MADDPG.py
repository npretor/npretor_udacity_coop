import random 
from collections import deque 
import torch 
from model import Actor, Critic

"""
A ddpg agent should have 

* Networks for the actor and critic, each with target and actual networks 

* Noise Class - standard or Oreson Ubleck or whatever 
* Replay buffer 

* soft copy function
* Step function
* Learn function 
* Reset function 
""" 


class MADDPG:
    def __init__(self, num_agents, observation_size, action_size, settings) -> None:
        # TODO: I'm missing something here about the output sizing and the action size probably
        self.target_actor = Actor(num_agents*observation_size, action_size, random.seed(0), settings)
        self.actual_actor = Actor(num_agents*observation_size, action_size, random.seed(0), settings) 

        # TODO: define critic 

        self.memory = ReplayBuffer(100) 
        self.settings = settings 
    
    def learn(self, experiences):
        """
        Get the experiences and do all the complated stuff here 


        """

        # TODO: calculate the reward and compare 
        self.settings['gamma'] * self.actual_actor 

        # Multiply the squared weights of each model to calculate the loss 

        # TODO: soft update as the last thing. From -> to
        self.soft_update(self.target_actor, self.actual_actor)
        self.soft_update(self.target_actor, self.actual_actor) 
        
        pass 


    def step(self, states, actions, next_states, dones):
        self.memory.add([states, actions, next_states, dones]) 

    def choose_action(self, state):
        """
        Explore-exploit ratio: 
        Noise ratio is about one in every hundred? 
        Add in noise falloff based on episodes here as well

        Two kinds of noise: 
        - Choose a random action with a given ratio
        - Apply noise to the action, a lot at first and falling off as the goal approaches
        """

        self.actual_actor.eval()
        action = self.actual_actor(state)
        # action is just for this agent 

        # 
        
        action * self.noise.sample() 

        return action 


class ReplayBuffer:
    def __init__(self, max_len=100) -> None:
        """
        Declare a deque of max_size 100 
        """
        self.samples = deque(max_len) 
        

    def add(self, item):
        """
        Add a sample to the top, remove one from the bottom 
        """
        self.samples.pop


class NoiseGen:
    def __init__(self, sample_size) -> None:
        pass

    def sample(self):
        """
        Random values in range between 0 and 1 of size sample_size 
        """
        return self.random