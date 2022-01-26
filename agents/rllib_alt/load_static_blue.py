import os
from pprint import pprint

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_rllib_alt import CybORGAgent, CustomModel

'''Aka the Volkswagen approach'''
class LoadBlueAgent:

    action_sequence = [18, 18, 18, 24, 37, 24, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 
                       17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17, 22, 17]

    step = 0

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        self.step = 0

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        self.step += 1
        return self.action_sequence[self.step-1]

    def reset(self):
        self.step = 0