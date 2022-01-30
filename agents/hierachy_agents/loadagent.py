import os
from pprint import pprint

import numpy as np
import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_hier import CustomModel, TorchModel
from agents.hierachy_agents.scaffold_env import CybORGScaffRM, CybORGScaffBL
from agents.hierachy_agents.hier_env import HierEnv
import os
from agents.hierachy_agents.sub_agents import sub_agents

class LoadBlueAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)
        relative_path = os.path.abspath(os.getcwd())
        print(relative_path)
        # load checkpoint locations of ech agent
        self.checkpoint = relative_path + '/log_dir/rl_controller_scaff/PPO_HierEnv_1e996_00000_0_2022-01-27_13-43-33/checkpoint_000212/checkpoint-212'
        self.BLcheckpoint_pointer = relative_path[:62] + sub_agents['B_line_trained']
        self.RMcheckpoint_pointer = relative_path[:62] + sub_agents['RedMeander_trained']

        #with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        #    self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file: {}".format(self.checkpoint))
        config = {
            "env": HierEnv,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
        }

        # Restore the controller model
        self.controller_agent = ppo.PPOTrainer(config=config, env=HierEnv)
        self.controller_agent.restore(self.checkpoint)
        self.observation = np.zeros((52*4))

        config["exploration_config"] ={
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }
            }
        #load agent trained against RedMeanderAgent
        config['env'] = CybORGScaffRM
        self.RM_def = ppo.PPOTrainer(config=config, env=CybORGScaffRM)
        self.RM_def.restore(self.RMcheckpoint_pointer)
        #load agent trained against B_lineAgent
        config['env'] = CybORGScaffBL
        self.BL_def = ppo.PPOTrainer(config=config, env=CybORGScaffBL)
        self.BL_def.restore(self.BLcheckpoint_pointer)



    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        #update sliding window
        obs = np.append(self.observation[52:], obs)
        self.observation = obs
        #select agent to compute action
        agent_to_select = self.controller_agent.compute_single_action(obs)
        if agent_to_select == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-52:])
        elif agent_to_select == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action = self.RM_def.compute_single_action(self.observation[-52:])
        return agent_action