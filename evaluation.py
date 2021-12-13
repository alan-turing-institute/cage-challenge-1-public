# Adapted from https://raw.githubusercontent.com/cage-challenge/cage-challenge-1/main/CybORG/CybORG/Evaluation/evaluation.py

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

from agents.helloworld_agent import TorchCustomModel as BasicAgent # Example
from agents.a2c.a2c_agent import Agent as A2CAgent
from agents.a2c.rollout import RolloutStorage
from agents.a2c.rnd import RunningMeanStd
from agents.ppo.PPO import PPO as PPOAgent
from agents_list import AGENTS
from config import configure

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
    # agent = AGENTS[args.name_of_agent](rnd=args.rnd, ) # BlueLoadAgent()

    print(f'Using agent {args.name_of_agent}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{args.name_of_agent}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {args.scenario}\n')
        data.write(f'author: {args.name}, team: {args.team}, technique: {args.name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n") # FIXME

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{args.scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {args.scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
                    
            if 'a2c' in args.name_of_agent:
                rollouts = RolloutStorage(steps=num_steps, processes=1, output_dimensions=action_space, 
                                            input_dimensions=wrapped_cyborg.observation_space.shape[0])

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    obs, rew, done, info = wrapped_cyborg.step(action)

                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                    if 'a2c' in args.name_of_agent:
                        int_reward = rew
                        masks = torch.FloatTensor([[0.0] if done else [1.0]])
                        bad_masks = torch.FloatTensor( [[0.0] if 'bad_transition' in info.keys() else [1.0]])
                        rollouts.insert(observation, rnn_states, action, action_log_prob, value, int_reward, masks, bad_masks)
                
                if 'a2c' in args.name_of_agent:
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
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')