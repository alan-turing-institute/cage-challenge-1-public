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
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.framework import try_import_tf, try_import_torch

# CybORG imports
from agents.hierachy_agents.hier_env import HierEnv
from typing import Any
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


tf1, tf, tfv = try_import_tf()

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

def normc_initializer(std: float = 1.0) -> Any:
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))

    return initializer

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)


    config = Trainer.merge_trainer_configs(
        DEFAULT_CONFIG,
        {
        "env": HierEnv,
        "env_config": {
            "null": 0,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "CybORG_hier_Model",
            #"vf_share_layers": False,
        },
        "lr": 0.0001,
        #"momentum": tune.uniform(0, 1),
        "num_workers": 4,  # parallelism
        "framework": "torch", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
    })

    stop = {
        "training_iteration": 5000,   # The number of times tune.report() has been called
        "timesteps_total": 10000000,   # Total number of timesteps
        "episode_reward_mean": -0.1, # When to stop.. it would be great if we could define this in terms
                                    # of a more complex expression which incorporates the episode reward min too
                                    # There is a lot of variance in the episode reward min
    }

    checkpoint = 'log_dir/PPO_3_step/PPO_HierEnv_ff1e1_00000_0_2022-01-31_16-08-15/checkpoint_000396/checkpoint-396'
   
    log_dir = 'log_dir/'

    analysis = tune.run(ppo.PPOTrainer, # Algo to use - alt: ppo.PPOTrainer, impala.ImpalaTrainer
                        config=config,
                        local_dir=log_dir,
                        stop=stop,
                        checkpoint_at_end=True,
                        checkpoint_freq=1,
                        restore=checkpoint,
                        keep_checkpoints_num=3,
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

