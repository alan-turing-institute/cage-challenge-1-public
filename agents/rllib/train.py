import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from RLlibWrapper import *

import sys
sys.path.append('cage-challenge-1/CybORG/')


from CybORG import CybORG
from CybORG.Agents.Wrappers import *
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents import B_lineAgent, GreenAgent, BlueMonitorAgent


################################### Training ###################################

def train():

    print("============================================================================================")

    env_name = "CybORG"
    print("Training environment name : " + env_name, flush=True)
    path = 'cage-challenge-1/CybORG/CybORG/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent, 'Green': GreenAgent})
    env = RLlibWrapper(ChallengeWrapper(env=cyborg, agent_name='Blue'))
    
    #cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent, 'Green': GreenAgent})
    #wrappers = FixedFlatWrapper(EnumActionWrapper(cyborg))
    #env = OpenAIGymWrapper(env=wrappers, agent_name='Blue')
    #env.reset()

    def make_env(_):
      print(_)
      return env
    register_env("train_envs_registered", make_env)
    

    obs = env.reset()
    print('Observation:',obs)
    print(73*'-')
    print('Action_Space:',env.action_space)
    print(73*'-')
    print('Observation Space:',env.observation_space)


    ################# Training ################
    #ray.init()
    config = {
        "env": "train_envs_registered",
        #"num_workers": 4,
        "num_gpus": 1,
        #"gamma": 0.4,
        #"num_gpus_per_worker": 0.25,
        #"num_envs_per_worker": 1,
        #"entropy_coeff": 0.01,
        #"num_sgd_iter": 10,
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        #"use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        #"use_gae": True,
        # The GAE (lambda) parameter.
        #"lambda": 1.0,
        # Initial coefficient for KL divergence.
        #"kl_coeff": 0.2,
        # Size of batches collected from each worker.
        #"rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        #"train_batch_size": 1000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        #"sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        #"shuffle_sequences": True,      
        #"vf_loss_coeff": 0.01,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        #"num_sgd_iter": 1000,
        #"lr": 0.0001,
        #"evaluation_interval": 100000,
        #"create_env_on_driver": True,
        #"framework": "torch",
    }
  
    stop = {
        "training_iteration": 10000,
        "timesteps_total": 10000000,
        "episode_reward_mean": 0.95,
    }


    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time, flush=True)
    results = tune.run("PPO", config=config, verbose=2)
    print(results)

    #env.close()




if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
    
