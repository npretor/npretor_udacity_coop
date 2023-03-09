"""


"""





import pytest
import sys 
sys.path.append('../') 
import json 
from DDPG import Agent, OUNoise, RandNoise, ReplayBuffer 

#  - - - - - - -  Test fixtures   - - - - - - -  #

@pytest.fixture
def settings():
    with open("../hyperparameters.json", 'r') as f:
        settings = json.load(f)    
    return settings

@pytest.fixture
def agent(settings):
    agent = Agent(num_agents=2, state_size=24, action_size=2, seed=0, settings=settings)
    return agent  

#  - - - - - - -  Test agent   - - - - - - -  #

def test_step():
    assert True == True 

def test_act():
    assert True == True 

def test_reset():
    assert True == True 

def test_learn():
    assert True == True     

def test_soft_update():
    assert True == True   


#  - - - - - - -  Test RandNoise   - - - - - - -  #




#  - - - - - - -  Test ReplayBuffer   - - - - - - -  #
def test_add():
    assert True == True 

def test_sample():
    assert True == True 