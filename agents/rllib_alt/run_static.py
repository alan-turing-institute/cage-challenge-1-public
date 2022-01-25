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
from load_static_blue import LoadBlueAgent

if __name__ == "__main__":

    # Restore the checkpointed model
    agent = LoadBlueAgent()


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
            action = agent.get_action(obs, None)
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
            
            print('Blue numerical moves ')
            for move in blue_move_numbers:
                print(move, end=', ')
            print()
            print('Number of episodes needed for max score: {}'.format(count_episodes))
            exit()

        episode_reward = 0
        done = False
        obs = env.reset()
        agent.reset()

        print('Initial environment state after reset: ')
        true_state = env.cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state,env.cyborg)
        print(true_table)