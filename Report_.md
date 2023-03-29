Questions to answer when starting out: 
- What can each agent see? 
- What is the agent? Both of the actors? Both of the critics? 
- How is credit assigned to an action? How many network have to be created? 

## What is the information flow through the program? 
### Flow level 1: 

1. The environment presents a state
1. Each actor observes it's local state, and infers an action 
2. The environment responds with a next_state, along with a reward
```
state -> action -> next_state, reward
```


Based on my reading of the <a href="https://arxiv.org/pdf/1706.02275.pdf">MA-DDPG paper</a>, the multi-agent version differs from single agent DDPG in a few ways. 
* The critic sees, saves, and learns all actions 
* The actor sees only it's perspective  

I decided to work on my previous DDPG code, and modify the agent and model to work as a multi-agent DDPG model.  
Principal differences: 
From Lowe et al.(2017): <i>"in which each agent trains a DDPG algorithm such that the actor with policy weights observes the local observations, while the critic is allowed to access the observations, actions, and the target policies of all agents in the training time. Then the critic of each agent concatenates all state-actions together as the input, and using the local reward obtains the corresponding Q-value. Either of the critics are trained by minimizing a DQN-like loss function." (shown below) </i>
![Action value function for MADDPG](./media/maddpg_action_value_function.png) 

Based on this I plan to modify the DDPG into a MADDPG using the following method. 


## Observation space
 - 24 vectors 

## Action space 
- Two actions, space of actions is -1 to 1, inclusive.  
    Moves are in 1 dimension: forwards or back, and jump: up or down. 
    Each action looks like:  [signed_move_direction, signed_jump_distance]
        actions: [lefthand_agent, righthand_agent] 

## Rewards 


## Dones


Questions 
- Q: Does the critic take all the states and actions and the actor just takes in the individual actions? The actor has to make a choice about an individual agent's actions, right? 
- A: After reading the paper in section 4.1: 
    <i> The critic is augmented with extra information about the policies of the other agents</i>
    Also the paper (a review of cooperative multi-agent deep reinforcement learning) explains that the agents act on their local observations and rewards, but the critic evaluates their actions based on the global actions  



# Learning function modifications 
1. Predict the next action by querying each target actor network for the next action 
2. Get the expected reward of those actions by sending the target critic network the all_next_actions and all_next_states  
3. Get the expected reward of the current (target network?) using: 
    y = agent_reward + gamma * (estimated_next_action_reward)
    Q_t = r_agent + (gamma * Q_t_next * (1 - dones)) 
4. Minimize the reward 
    Also the paper (a review of cooperative multi-agent deep reinforcement learning) explains that the agents act on their local observations and rewards, but the critic evaluates their actions based on the global actions 



## Information flow 

## Observe, act, learn, repeat 

1. Observe.         state[24]
2. Act              action[2], reward[1], next_state[24], done[1]
3. Each agent's info is sent back from the simulation. There are two agents i end up with data structured like this:
    state = np.ndarray of shape         2, 24
    action = np.ndarray of shape        2, 2 
    reward = np.ndarray of shape        2, 1
    next_state = np.ndarray of shape    2, 24
    done = np.ndarray of shape          2, 1
                    state[24], action[2], reward[1], next_state[24], done[1]
   Feed the experience information into a deque. Each variable fed in must be 1 dimensional, but we need to save both agent's data. My solution now is to do a np.flatten / reshape(-1) on each experience. 


# Log 

* Trained 
* Added batch normalization. Also realized I was not deep copying at initialization, moved from agentorchestrator to ddpg 



1. Observe state  Numpy[2,24]
2. Choose a