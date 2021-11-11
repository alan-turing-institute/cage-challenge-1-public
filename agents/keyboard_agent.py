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

cyborg = CybORG(path, 'sim',agents={'Blue':BlueMonitorAgent})
env = RedTableWrapper(env=cyborg, output_mode='table')

agent = KeyboardAgent()

results = env.reset('Red')
obs = results.observation
action_space = results.action_space

for i in range(3):
    print(obs)
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Red')
    obs = results.observation