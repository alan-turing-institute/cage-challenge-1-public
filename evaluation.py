# Adapted from https://raw.githubusercontent.com/cage-challenge/cage-challenge-1/main/CybORG/CybORG/Evaluation/evaluation.py

import inspect
import time, sys
from statistics import mean, stdev

sys.path.append("cage-challenge-1/CybORG")


from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table

from agents_list import AGENTS
from agents.helloworld_agent import TorchCustomModel as BasicAgent # Example
from config import configure
import os

MAX_EPS = 10
agent_name = 'Blue'

def wrap(env):
    return OpenAIGymWrapper(agent_name,
                            EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))


if __name__ == "__main__":
    cyborg_version = '1.2'
    args = configure()
    # FIXME what is this for?
    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # # Change this line to load your agent
    agent = AGENTS[args.name_of_agent]()
    #agent.load(os.path.abspath(os.getcwd())+'/agents/a2c/saved_models/a2c/actor_critic.pt') # BlueLoadAgent()

    print(f'Using agent {args.name_of_agent}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{args.name_of_agent}.txt'
    table_file = time.strftime("%Y%m%d_%H%M%S") + f'_table_file.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {args.scenario}\n')
        data.write(f'author: {args.name}, team: {args.team}, technique: {args.name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n") # FIXME

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{args.scenario}.yaml'

    agent_args = args
    name_of_agent = args.name_of_agent

    print("agent_args", agent_args)

    print(f'using CybORG v{cyborg_version}, {args.scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = ChallengeWrapper(env=cyborg, agent_name='Blue') #wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)

            if args.show_table:
                print('Adversary: {}'.format(red_agent))
            #agent = AGENTS[args.name_of_agent](args).load(os.path.abspath(os.getcwd())+'/agents/a2c/saved_models/a2c/actor_critic.pt')

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                if args.show_table:
                    print('Episode: {}'.format(i))
                # cyborg.env.env.tracker.render()
                moves = []
                successes = []
                tables = []
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)

                    if args.show_table:
                        #print('Step: {}'.format(j))
                        red_move = wrapped_cyborg.get_last_action('Red').__str__()
                        blue_move = wrapped_cyborg.get_last_action('Blue').__str__()
                        green_move = wrapped_cyborg.get_last_action('Green').__str__()
                        #print('Blue: {}, Green: {}, Red: {}'.format(blue_move, green_move, red_move))
                        #print('Blue: {}, Green: {}, Red: {}'.format(blue_move, green_move, red_move))
                        true_state = cyborg.get_agent_state('True')
                        true_table = true_obs_to_table(true_state,cyborg)
                        #print(true_table)
                        success_observation = wrapped_cyborg.get_attr('environment_controller').observation
                        blue_success = success_observation['Blue'].action_succeeded
                        red_success = success_observation['Red'].action_succeeded
                        green_success = success_observation['Green'].action_succeeded
                        moves.append((blue_move, green_move, red_move))
                        successes.append((blue_success, green_success, red_success))
                        tables.append(true_table)
                    r.append(rew)


                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                if i == 0 and args.show_table == True:
                    with open(table_file, 'a+') as table_out:
                        table_out.write('\nRed Agent: {}, Num_steps: {}'.format(red_agent.__name__, num_steps.__str__()))
                        for move in range(len(moves)):
                            table_out.write('\nStep: {}, Blue: {}, Green: {}, Red: {}\n'.format(move, moves[move][0], moves[move][1], moves[move][2]))
                            table_out.write('Success Blue: {}, Success Green: {}, Success Red: {}\n'.format(successes[move][0], successes[move][1], successes[move][2]))
                            table_out.write('Blue Reward: {}\n'.format(r[move]))
                            table_out.write(str(tables[move]))

                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')