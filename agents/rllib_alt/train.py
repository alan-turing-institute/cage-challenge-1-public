"""Alternative RLLib model based on local training

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
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
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

# CybORG imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()

parser.add_argument("--run", type=str, default="PPO",
    help="The RLlib-registered algorithm to use.")

parser.add_argument("--framework", choices=["tf", "tf2", "tfe", "torch"], default="tf",
    help="The DL framework specifier.")

parser.add_argument("--as-test", action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
    
parser.add_argument("--stop-iters", type=int, default=50,
    help="Number of iterations to train.")

parser.add_argument("--stop-timesteps", type=int, default=100,
    help="Number of timesteps to train.")

parser.add_argument("--stop-reward", type=float, default=0.1,
    help="Reward at which we stop training.")


class CybORGAgent(gym.Env):

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agents = {
            'Red': B_lineAgent,
            'Green': GreenAgent
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
        if self.steps == 20:
            return result[0], result[1], True, result[3]
        assert(self.steps<=20)
        return result
    
    def seed(self, seed=None):
        random.seed(seed)


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1, ), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(config.worker_index * config.num_workers)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return [self.cur_pos], \
            random.random() * 2 if done else -0.1, done, {}

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
    args = parser.parse_args()
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", CustomModel
        if args.framework == "tf" else exit())

    config = {
        "env": CybORGAgent,  # or "corridor" if registered above
        "env_config": {
            "corridor_length": 5,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
