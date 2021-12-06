import numpy as np
from gym import spaces, Env
from typing import Union, List
from prettytable import PrettyTable

import sys,os
print(os.getcwd())
sys.path.append('cage-challenge-1/CybORG/')
sys.path.append('/content/drive/MyDrive/cage-challenge-1/cage-challenge-1/CybORG/')


from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper


class RLlibWrapper(BaseWrapper):
    def __init__(self, env: BaseWrapper):
        super().__init__(env)
        self.steps = 1
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space        
        self.observation_space = self.env.observation_space
        #self.reward_range = self.env.reward_range
        #self.metadata = self.env.metadata
        self.action = None #self.evn.action

    def step(self,action=None):
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == 20:
            return result[0], result[1], True, result[3]
        assert(self.steps<=20)
        return result
        

    def reset(self, agent=None):
        self.steps = 1
        return self.env.reset()

    def render(self):
        return self.env.render()

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self.env.get_attr('get_agent_state')(agent)

    def get_action_space(self,agent):
        return self.env.get_action_space(agent)

    def get_last_action(self,agent):
        return self.env.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.env.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.env.get_attr('get_rewards')()
