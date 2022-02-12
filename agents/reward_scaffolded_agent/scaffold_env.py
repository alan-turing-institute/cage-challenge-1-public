import os
import random
import inspect
import gym
import ray
import re
from ray import tune

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from ray.rllib.env.env_context import EnvContext

class CybORGScaff(gym.Env):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
        'Red': B_lineAgent  # , #RedMeanderAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        self.cyborg = CybORG(self.path, 'sim', agents=self.agents)

        #self.env = OpenAIGymWrapper('Blue',
        #                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(self.cyborg))))
        self.env  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.steps = 1
        self.agent_name = self.env.agent_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action = None

    def reset(self):
        self.steps = 1
        return self.env.reset()

    def step(self, action=None):
        result = self.env.step(action=action)
        if str(self.env.get_last_action('Red')) and re.search('ExploitRemoteService' , str(self.env.get_last_action('Red')), re.I):
            observation, reward, done, info = result
            target_ip = re.search('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', str(self.env.get_last_action('Red'))).group()
            for host, ip in self.env.get_ip_map().items():
                if str(ip) == target_ip:
                    if re.search('Enterprise',host, re.I):
                        reward += -0.5
                    elif re.search('Op_Server',host, re.I):
                        reward += -0.5
                    elif re.search('Op_Host',host, re.I):
                        reward += -0.05
                    elif re.search('User',host, re.I):
                        reward += -0.05
            result = observation, reward, done, info
        self.steps += 1
        if self.steps == 100:
            return result[0], result[1], True, result[3]
        assert (self.steps <= 100)
        return result

    def seed(self, seed=None):
        random.seed(seed)
