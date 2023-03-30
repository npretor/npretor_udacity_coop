# npretor_udacity_coop
Cooperative deep reinforcement learning project for the Udacity Nanodegree

## Install 
1. Install conda 
2. Setup environment 

```
conda activate drlnd 
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
```

3. Fix an issue with Pytorch
<b>Change the pytorch version in the requirements file to be the latest version, or remove version number altogether </b>
4. Install requirements 
```
pip install .
```

5. Create an IPython kernel for the drlnd environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

### Run tests
```
cd tests 
pytest 
```


## Train 
```
conda activate drlnd 
wandb disabled 
cd $YOUR_REPO_INSTALL_LOCATION
python3 train.py 
```

## Demo 
```
wandb disabled
conda activate drlnd 
python3 demo.py 
``` 

### Project environment 
The environment is composed of two agents with a ball and a net. 
<b> State space</b> The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket.
<b> Action space </b>Each agent has two actions, space of actions is -1 to 1, inclusive. Moves are in 1 dimension: forwards or back, and jump: up or down. Each action looks like:  
> [signed_move_direction, signed_jump_distance]
        actions: [lefthand_agent, righthand_agent] 

<b> Solution </b> The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5. When training I had the model save the weights if it went over the threshold of 0.51, then set the goal to be 0.1 higher and kept training. I reached a value of 0.7 before the training started to level off 

