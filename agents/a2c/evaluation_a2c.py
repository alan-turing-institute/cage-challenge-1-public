import inspect
import time
from statistics import mean, stdev
import torch
import numpy as np
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
from a2c.a2c_agent import AgentA2C
from a2c.rollout import RolloutStorage
from a2c.rnd.rnd import RunningMeanStd


def wrap(env, agent_name):
    return OpenAIGymWrapper(agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))


def evaluate(r_mean, obs_rms, agent=BlueLoadAgent()):
    MAX_EPS = 10
    agent_name = 'Blue'
    cyborg_version = '1.0'

    scenario = 'Scenario1b'
    # ask for a name
    name = 'm'#input('Name: ')
    # ask for a team
    team = 'turing'#input("Team: ")
    # ask for a name for the agent
    name_of_agent = 'maxss'#input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    #agent = BlueLoadAgent()



    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'



    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg, agent_name)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)

            rollouts = RolloutStorage(steps=num_steps, processes=1, output_dimensions=action_space, input_dimensions=wrapped_cyborg.observation_space.shape[0])

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    with torch.no_grad():
                        value, action, action_log_prob, rnn_states = agent.actor_critic.act(
                            rollouts.observations[j], rollouts.rnn_states[j],
                            rollouts.masks[j])                    
                    obs, rew, done, info = wrapped_cyborg.step(action)

                    """intrinsic_reward = agent.rnd.compute_intrinsic_reward((obs - obs_rms.mean) / np.sqrt(obs_rms.var)).detach().numpy()
                    mean, std, count = np.mean(intrinsic_reward), np.std(intrinsic_reward), len(intrinsic_reward)
                    r_mean.update_from_moments(mean, std ** 2, count)
                    intrinsic_reward /= np.sqrt(r_mean.var)
                    intrinsic_reward = np.clip(intrinsic_reward, -0.1, 0.1)
                    obs_rms.update(obs)
                    int_reward = rew + intrinsic_reward"""
                    int_reward = rew

                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                    masks = torch.FloatTensor([[0.0] if done else [1.0]])
                    bad_masks = torch.FloatTensor( [[0.0] if 'bad_transition' in info.keys() else [1.0]])
                    rollouts.insert(observation, rnn_states, action, action_log_prob, value, int_reward, masks, bad_masks)
                with torch.no_grad():
                    next_value = agent.actor_critic.get_value(rollouts.observations[-1],
                                                      rollouts.rnn_states[-1],
                                                      rollouts.masks[-1]).detach()
                rollouts.compute_returns(next_value, gamma=0.99)
                value_loss, action_loss, dist_entropy = agent.update(rollouts)
                rollouts.after_update()

                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {np.mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {np.mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')



path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
cyborg = CybORG(path, 'sim')
wrapped_cyborg = wrap(cyborg, 'Blue')
action_space = wrapped_cyborg.action_space.n
obs_space    = wrapped_cyborg.observation_space.shape[0]
r_mean = RunningMeanStd()
obs_rms = RunningMeanStd(shape=(1, obs_space))
print('Initialise observation standardisation...')
for step in range(10 * 100):
    actions = np.array([np.random.randint(0, action_space) for i in range(1)])[0]
    obs, _, _, _ = wrapped_cyborg.step(actions)

    obs_rms.update(obs)
agent = AgentA2C(rnd=False, action_space=action_space, processes=1, input_space=obs_space)
agent.load_model()

evaluate(agent=agent, obs_rms=obs_rms, r_mean=r_mean)