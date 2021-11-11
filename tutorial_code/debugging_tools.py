import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Shared.Actions import Restore
# Converts true_state to human-readable table
from CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table 

# In order to help users understand what is going on, it is necessary to be
# able to pull out the true state of the network at any time. This is obtained
# by calling the get_agent_state method and passing in 'True'. 
# Since this observation is huge, we will focus on 'User0'.
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

env = CybORG(path, 'sim',agents={'Red':B_lineAgent})

results = env.reset(agent='Blue')

true_state = env.get_agent_state('True') # Pull out true state of network

#pprint(true_state['User0'])
#print(76*'-')

true_table = true_obs_to_table(true_state,env)
print(true_table)

for i in range(3):
    env.step()
    true_state = env.get_agent_state('True')
    true_table = true_obs_to_table(true_state,env)
    print(env.get_last_action('Red'))
    print(true_table)
    print(76*'-')

print('Rewards for all agents:')
pprint(env.get_rewards())

# Blue team then restores 'User1' and we can see Red's access is gone.
action = Restore(hostname='User1',session=0,agent='Blue')
env.step(action=action,agent='Blue')

true_state = env.get_agent_state('True')
true_table = true_obs_to_table(true_state,env)
print(true_table)

# CybORG has a host of other tools to help understand the agent state. We have already see the get_observation method.
env.reset()
env.step()

red_obs = env.get_observation('Red')
print('Red Obvs:')
pprint(red_obs)
print('Last Red Action:')
last_red_action = env.get_last_action('Red')
print(last_red_action)


blue_obs = env.get_observation('Blue')
print('Blue Obvs:')
pprint(red_obs)
pprint(blue_obs)
print('Last Blue Action:')
last_blue_action = env.get_last_action('Blue')
print(last_blue_action)

print(76*'-')

print('Red Action Space:')
red_action_space = env.get_action_space('Red')
print(list(red_action_space))

print('Blue Action Space:')
blue_action_space = env.get_action_space('Blue')
print(list(blue_action_space.keys()))


print('Hostnames <-> IPs (known only to red):')
pprint(env.get_ip_map())

print('Rewards for all agents:')
pprint(env.get_rewards())

# env.set_seed(100)