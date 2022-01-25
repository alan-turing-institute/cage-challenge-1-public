"""Alternative RLLib model based on local training

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import inspect

# Ray imports
import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.schedulers import ASHAScheduler # https://openreview.net/forum?id=S1Y7OOlRZ algo for early stopping
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

# CybORG imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent, BlueMonitorAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

tf1, tf, tfv = try_import_tf()


class CybORGAgent(gym.Env):

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
            'Red': B_lineAgent#, #RedMeanderAgent, 'Green': GreenAgent
    }

    """The CybORGAgent env"""
    def __init__(self, config: EnvContext):

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
        result = self.env.step(action=action)
        self.steps += 1
        if self.steps == 100:
            return result[0], result[1], True, result[3]
        assert(self.steps<=100)
        return result
    
    def seed(self, seed=None):
        random.seed(seed)


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)

        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_DQN_Model", CustomModel)

    config = Trainer.merge_trainer_configs(
        APEX_DEFAULT_CONFIG,
        {
            "env": CybORGAgent,  
            
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "model": {
                "custom_model": "CybORG_DQN_Model",
                "vf_share_layers": True,
            },

            "framework": "tf2", # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)

            # === Settings for Rollout Worker processes ===
            "num_workers": 4,  # No. rollout workers for parallel sampling.
            
            # === Settings for the Trainer process ===  
            "lr": 1e-4,     

            # === Environment settings ===
            #"preprocessor_pref": "deepmind",
        
            # === DQN/Rainbow Model subset config ===
            "num_atoms": 1,     # Number of atoms for representing the distribution of return. 
                                # Use >1 for distributional Q-learning (Rainbow config)
                                # 1 improves faster than 2
            "v_min": -1000.0,   # Minimum Score
            "v_max": -0.0,      # Set to maximum score
            "noisy": True,      # Whether to use noisy network (Set True for Rainbow)
            "sigma0": 0.5,      # control the initial value of noisy nets
            "dueling": True,    # Whether to use dueling dqn
            "hiddens": [256],   # Dense-layer setup for each the advantage branch and the value
                                # branch in a dueling architecture.
            "double_q": True,   # Whether to use double dqn
            "n_step": 3,        # N-step Q learning (Out of 1, 3 and 6, 3 seems to do learn most quickly)

            "learning_starts": 100, # Number of steps of the evvironment to collect before learing starts
            
        }
    )

    stop = {
        "training_iteration": 400,   # The number of times tune.report() has been called
        "timesteps_total": 8000000,   # Total number of timesteps
        "episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    log_dir = 'log_dir/'

    analysis = tune.run(dqn.ApexTrainer, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config, 
                        local_dir=log_dir,
                        stop=stop,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        keep_checkpoints_num=2,
                        checkpoint_score_attr="episode_reward_mean")

    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )

    checkpoint_pointer.write(last_checkpoint)
    print("Best model checkpoint written to: {}".format(last_checkpoint))

    # If you want to throw an error
    #if True:
    #    check_learning_achieved(analysis, 0.1)

    checkpoint_pointer.close()
    ray.shutdown()

    # You can run tensorboard --logdir=log_dir/PPO... to visualise the learning processs during and after training

