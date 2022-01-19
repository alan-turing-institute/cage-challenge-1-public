import os
from pprint import pprint

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train import CybORGAgent, CustomModel

if __name__ == "__main__":

    ModelCatalog.register_custom_model("my_model", CustomModel)

    with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        checkpoint_pointer = chkpopfile.readlines()[0]
    print("Using checkpoint file: {}".format(checkpoint_pointer))

    config = {
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "num_workers": 4,  # parallelism
        "framework": "tf",
    }

    # Restore the checkpointed model
    agent = ppo.PPOTrainer(config=config, env=CybORGAgent)
    agent.restore(checkpoint_pointer)

    # Run the model...
    env = CybORGAgent(EnvContext)

    episode_reward = 0
    done = False
    obs = env.reset()

    true_state = env.cyborg.get_agent_state('True')
    true_table = true_obs_to_table(true_state,env.cyborg)
    print(true_table)

    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        print(76*'-')

        print('Blue Action: {}'.format(action))
        print('Reward: {}, Episode reward: {}'.format(reward, episode_reward))
        print('Network state:')
        true_state = env.cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state,env.cyborg)
        print(true_table)
        print(76*'-')
        print('.')