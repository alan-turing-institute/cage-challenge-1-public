import os
from pprint import pprint

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_rllib_alt import CybORGAgent, CustomModel

class LoadBlueAgent:

    """
    Load the agent model using the latest checkpoint and return it for evaluation
    """
    def __init__(self) -> None:
        ModelCatalog.register_custom_model("my_model", CustomModel)

        with open ("checkpoint_pointer.txt", "r") as chkpopfile:
            self.checkpoint_pointer = chkpopfile.readlines()[0]
        print("Using checkpoint file: {}".format(self.checkpoint_pointer))

        config = {
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "my_model",
                "vf_share_layers": True,
            },
            "num_workers": 4,  # parallelism
            "framework": "tf2",
        }

        # Restore the checkpointed model
        self.agent = dqn.ApexTrainer(config=config, env=CybORGAgent)
        self.agent.restore(self.checkpoint_pointer)

    """Compensate for the different method name"""
    def get_action(self, obs, action_space):
        return self.agent.compute_single_action(obs)