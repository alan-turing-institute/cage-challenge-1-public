# The KeyboardAgent allows a human user to manually choose actions. 
# This is useful for getting an intuition for the scenario.
import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Shared.Actions import *
from CybORG.Agents.Wrappers import *

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

cyborg = CybORG(path, 'sim',agents={'Red':B_lineAgent})
env = BlueTableWrapper(env=cyborg, output_mode='table')

agent = KeyboardAgent()

results = env.reset('Blue')
obs = results.observation
action_space = results.action_space
episode_reward = 0

for i in range(10):
    print(obs)
    #print(env.get_table())
    print(env.get_rewards())
    action = agent.get_action(obs,action_space)
    results = env.step(action=action, agent='Blue')
    obs = results.observation
    episode_reward+=results.reward

print('Episode reward: {}'.format(episode_reward))