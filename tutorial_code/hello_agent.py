import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Shared.Actions import *

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

env = CybORG(path,'sim')

# Scenario 1b has two rules-based Red Agents. The first is our good 
# friend the B_lineAgent. This represents an actor who has inside 
# information, so is able to beeline straight towards the OpServer.

agent = B_lineAgent()

results = env.reset('Red')
obs = results.observation
action_space = results.action_space

for i in range(16):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Red')
    obs = results.observation
    
    print(action)
    print('Rewards for all agents:')
    pprint(env.get_rewards())

print(76*'-')

# This agent runs along a predetermined path to the Op_Server, 
# but is smart enough able to recover its position if interrupted. 
# We can see below after Blue Team restores some hosts, the agent 
# works out where the error in and re-exploits its way to the Op_Server.

action = Restore(hostname='Op_Server0',session=0,agent='Blue')
env.step(action=action,agent='Blue')

action = Restore(hostname='Enterprise2',session=0,agent='Blue')
env.step(action=action,agent='Blue')

action = Restore(hostname='Enterprise1',session=0,agent='Blue')
env.step(action=action,agent='Blue')

for i in range(12):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Red')
    obs = results.observation
            
    print(action)
    print('Success:',obs['success'])

print(76*'-')

# The other red agent is the MeanderAgent. This performs a breadth first 
# search on all known hosts, scanning each one in turn, before attempting
#  a mix of exploit and privilege escalate on the rest. 
# This is an extremely slow agent in contrast to the laser-focussed B_lineAgent.

agent = RedMeanderAgent()

results = env.reset('Red')
obs = results.observation
pprint(obs)
exit()
action_space = results.action_space

for i in range(46):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Red')
    obs = results.observation
    
    print(action)

print(76*'-')

# The Meander Agent is also able to recover from Blue's disruption.
action = Restore(hostname='Op_Server0',session=0,agent='Blue')
env.step(action=action,agent='Blue')

action = Restore(hostname='Enterprise2',session=0,agent='Blue')
env.step(action=action,agent='Blue')

action = Restore(hostname='Enterprise1',session=0,agent='Blue')
env.step(action=action,agent='Blue')

action = Restore(hostname='Enterprise0',session=0,agent='Blue')
env.step(action=action,agent='Blue')

for i in range(24):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Red')
    obs = results.observation
    print(env.get_last_action('Red'))
    print('Success:',obs['success'])

# The BlueReactRemoveAgent will wait until it sees suspicious activity, 
# before using remove on all the hosts it has flagged. However, due to 
# the 5% change that Red's exploit is missed, Red will always eventually 
# get to the Op_Server.

print(76*'-')

env = CybORG(path,'sim',agents={'Red':B_lineAgent})

agent = BlueReactRemoveAgent()

results = env.reset('Blue')
obs = results.observation
action_space = results.action_space

for i in range(12):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Blue')
    obs = results.observation
    print(env.get_last_action('Blue'))

print(76*'-') 

# The BlueReactRestoreAgent is the same as the React agent above, but 
# uses the Restore action.

agent = BlueReactRestoreAgent()

results = env.reset('Blue')
obs = results.observation
action_space = results.action_space

for i in range(12):
    action = agent.get_action(obs,action_space)
    results = env.step(action=action,agent='Blue')
    obs = results.observation
    print(env.get_last_action('Blue'))

print(76*'-') 

# An important part of CybORG Scenario1b is the Green agent, which represents 
# the users on the network. The Green Agent is very simple, it only performs 
# a scanning action on random hosts some of the time. 
# This is only visible by Blue Agent.

agent = GreenAgent()

results = env.reset('Green')
obs = results.observation
action_space = results.action_space

for i in range(12):
    print(agent.get_action(obs,action_space))