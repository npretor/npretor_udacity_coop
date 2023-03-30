
# Information flow 

## Load environment 
    All_states      [2, 24] 
    


## Choose action 


## Render response   
    All_states      [2, 24]
    All_dones       [2]
    All_rewards     [2] 
    All_actions     [2, 2]
    All_next_states [2,24]   
 
## Add to memory 
* Must be 1D arrays to add to memory. 
    * Transmute to [agent1, agent2, ...agentn] 
    All_states      [1, 48] 
    All_dones       [2] 
    All_rewards     [2]  
    All_actions     [1, 4] 
    All_next_states [1, 48]        

## Learn 
* Remove from memory and reshape
* The model needs a batch of memories of shape [batch_size, state_size] 
    All_states      [2, 24]
    All_dones       [2]
    All_rewards     [2] 
    All_actions     [2, 2]
    All_next_states [2,24]    


## - - - Critic model definition - - - - # 
Critic takes in the agent's action[2], and all the states [48]

L1: [state_size*num_agents=48]
L2: [2+128] or [action + layer_size]

In order to train, we need to 
