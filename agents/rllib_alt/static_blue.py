import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import inspect

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

class CybORGAgent_Glory(gym.Env):

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
            'Red': B_lineAgent
    }

    # Screamning intensifies
    action_sequence = [18, 18, 18, 24, 37, 24, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17]

    """The CybORGAgent env"""
    def __init__(self):

        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)
        self.env = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.steps = 1
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space        
        self.observation_space = self.env.observation_space
        self.action = None
    
    def reset(self):
        self.steps = 1
        return self.env.reset()
    
    def step(self, action=None):
        action = action[self.steps-1]
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == 100:
            return result[0], result[1], True, result[3]
        assert(self.steps<=100)
        return result
    
    def seed(self, seed=None):
        random.seed(seed)