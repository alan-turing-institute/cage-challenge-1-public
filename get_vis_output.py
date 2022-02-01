import inspect
import time
from statistics import mean, stdev

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

from agents.hierachy_agents.loadagent import LoadBlueAgent

MAX_EPS = 10
agent_name = 'Blue'

def wrap(env):
    return OpenAIGymWrapper(env=env, agent_name='Blue')


if __name__ == "__main__":
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    # ask for a name
    name = 'Mindrake ' #input('Name: ')
    # ask for a team
    team = 'Mindrake' #input("Team: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = LoadBlueAgent()

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    rew_total = 0

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    blue_moves = []
    blue_move_numbers = []
    red_moves = []
    green_moves = []
    table_file = 'visualisation/logs_to_vis/results.txt'
    with open(table_file, 'w+') as table_out:
        table_out.write('\n')
    redAgent = B_lineAgent


    cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    wrapped_cyborg = ChallengeWrapper(env=cyborg, agent_name='Blue') #wrap(cyborg)

    observation = wrapped_cyborg.reset()

    action_space = wrapped_cyborg.get_action_space(agent_name)
    moves = []
    successes = []
    tables = []
    total_reward = 0
    actions = []
    for j in range(100):
        action = agent.get_action(observation, action_space)
        observation, rew, done, info = wrapped_cyborg.step(action)

        blue_moves += [info['action'].__str__()]
        blue_move_numbers += [action]
        red_moves += [wrapped_cyborg.get_last_action('Red').__str__()]

        green_moves += [wrapped_cyborg.get_last_action('Green').__str__()]

        red_move = wrapped_cyborg.get_last_action('Red').__str__()
        blue_move = wrapped_cyborg.get_last_action('Blue').__str__()
        green_move = wrapped_cyborg.get_last_action('Green').__str__()
        true_state = cyborg.get_agent_state('True')
        true_table = true_obs_to_table(true_state, cyborg)
        success_observation = wrapped_cyborg.get_attr('environment_controller').observation
        blue_success = success_observation['Blue'].action_succeeded
        red_success = success_observation['Red'].action_succeeded
        green_success = success_observation['Green'].action_succeeded
        moves.append((blue_move, green_move, red_move))
        successes.append((blue_success, green_success, red_success))
        tables.append(true_table)
        total_reward += rew




    with open(table_file, 'a+') as table_out:
        table_out.write('\nRed Agent: B_lineAgent, Num_steps: 100')
        for move in range(len(moves)):
            table_out.write('\nBlue Action: {}\n'.format(moves[move][0]))
            table_out.write('Reward: {}, Episode reward: {}\n'.format(rew, total_reward))
            table_out.write('Network state:\n')
            table_out.write(str(tables[move]))
            table_out.write('\n.\n\n')
