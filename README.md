# npretor_udacity_coop
Cooperative deep reinforcement learning project for the Udacity Nanodegree

## Install 
```
TODO 
```

### Run tests
```
cd tests 
pytest 
```


## Train 
```
conda activate drlnd 
python3 train.py 
```

## Demo 
```
conda activate drlnd 
python3 demo.py 
``` 

### Project environment 
The environment is composed of two agents with a ball and a net. 
<b> State space</b> The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket.
<b> Action space </b>Each agent has two actions, space of actions is -1 to 1, inclusive. Moves are in 1 dimension: forwards or back, and jump: up or down. Each action looks like:  
> [signed_move_direction, signed_jump_distance]
        actions: [lefthand_agent, righthand_agent] 

<b> Solution </b> The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.