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
        "eager_tracing": True,
        "num_workers": 4,  # parallelism
        "framework": "tf2",
        "lr": 0.0001
    }

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

    while True:
        blue_moves = []
        blue_move_numbers = []
        red_moves = []
        green_moves = []
        

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
        if episode_reward >= -1.3:
            print('episode reward: {}'.format(episode_reward))
            print('Gameplay step through:')
            
            for move_idx, move in enumerate(zip(blue_moves, green_moves, red_moves)):
                print('{}. Blue: {}, Green: {}, Red: {}'.format(move_idx, move[0], move[1], move[2]))
            
            #print('Blue numerical moves ')
            #for move in moves:
            #    print(move, end=', ')
            #print()
            exit()

        episode_reward = 0
        done = False
        obs = env.reset()

        print('Initial environment state after reset: ')
        true_state = env.cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state,env.cyborg)
        print(true_table)