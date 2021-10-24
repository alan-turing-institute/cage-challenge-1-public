import inspect
import random
from pprint import pprint
from CybORG import CybORG

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

env = CybORG(path, 'sim')

results = env.reset(agent='Red')
obs = results.observation
pprint(obs)