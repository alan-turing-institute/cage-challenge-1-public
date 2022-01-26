import os
from pprint import pprint

import ray
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env.env_context import EnvContext
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

from CybORG import CybORG
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from train_rllib_alt import CybORGAgent, CustomModel

if __name__ == "__main__":

    ModelCatalog.register_custom_model("CybORG_DQN_Model", CustomModel)

    with open ("checkpoint_pointer.txt", "r") as chkpopfile:
        checkpoint_pointer = chkpopfile.readlines()[0]
    print("Using checkpoint file: {}".format(checkpoint_pointer))

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

    # Restore the checkpointed model
    agent = dqn.ApexTrainer(config=config, env=CybORGAgent)
    agent.restore(checkpoint_pointer)


    # Run the model...
    env = CybORGAgent(EnvContext)

    episode_reward = 0
    done = False
    obs = env.reset()

    print('Initial environment state: ')
    true_state = env.cyborg.get_agent_state('True')
    true_table = true_obs_to_table(true_state,env.cyborg)
    print(true_table)

    count_episodes = 0

    while True:
        blue_moves = []
        blue_move_numbers = []
        red_moves = []
        green_moves = []
        count_episodes += 1


        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            blue_moves += [info['action'].__str__()]
            blue_move_numbers += [action]
            red_moves += [env.env.get_last_action('Red').__str__()]

            green_moves += [env.env.get_last_action('Green').__str__()]

            #print('Blue Action: {}'.format(action))
            #print('Reward: {}, Episode reward: {}'.format(reward, episode_reward))
            #print('Network state:')

            #true_state = env.cyborg.get_agent_state('True')
            #true_table = true_obs_to_table(true_state,env.cyborg)
            #print(true_table)
            #print('.')
        print('\n')
        if episode_reward >= -5.0:
            print('episode reward: {}'.format(episode_reward))
            print('Gameplay step through:')
            
            for move_idx, move in enumerate(zip(blue_moves, green_moves, red_moves)):
                print('{}. Blue: {}, Green: {}, Red: {}'.format(move_idx, move[0], move[1], move[2]))
            
            print('Blue numerical moves ')
            for move in blue_move_numbers:
                print(move, end=', ')
            print()
            print('Number of episodes needed for max score: {}'.format(count_episodes))
            exit()

        episode_reward = 0
        done = False
        obs = env.reset()

        print('Initial environment state after reset: ')
        true_state = env.cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state,env.cyborg)
        print(true_table)