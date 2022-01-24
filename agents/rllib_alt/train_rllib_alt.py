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
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

# CybORG imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, GreenAgent, BaseAgent, RedMeanderAgent
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

    # Can also register the env creator function explicitly with:
    # register_env("env name", lambda config: EnvClass(config))
    ModelCatalog.register_custom_model("my_model", CustomModel)

    config = {
        "env": CybORGAgent,  
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr": grid_search([1e-4]),  
        "num_workers": 4,  # parallelism

        "framework": "tf2", # May also use "tf2", "tfe" or "torch" if supported
        "eager_tracing": True, # In order to reach similar execution speed as with static-graph mode (tf default)
        #"vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO       

        # DQN/Rainbow config
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -1000.0,
        "v_max": -0.1,  # Set to maximum score
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": True,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,

        # === Prioritized replay buffer ===
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.4,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,

        # Callback to run before learning on a multi-agent batch of
        # experiences.
        "before_learn_on_batch": None,

        # The intensity with which to update the model (vs collecting samples
        # from the env). If None, uses the "natural" value of:
        # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
        # `num_envs_per_worker`).
        # If provided, will make sure that the ratio between ts inserted into
        # and sampled from the buffer matches the given value.
        # Example:
        #   training_intensity=1000.0
        #   train_batch_size=250 rollout_fragment_length=1
        #   num_workers=1 (or 0) num_envs_per_worker=1
        #   -> natural value = 250 / 1 = 250.0
        #   -> will make sure that replay+train op will be executed 4x as
        #      often as rollout+insert op (4 * 250 = 1000).
        # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further
        # details.
        "training_intensity": None,

        # === Parallelism ===
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        
    }

    stop = {
        "training_iteration": 1000,   # The number of times tune.report() has been called
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
                        checkpoint_freq=2,
                        keep_checkpoints_num=2,
                        checkpoint_score_attr="reward")

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

