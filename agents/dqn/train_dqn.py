import time
from matplotlib import pyplot as plt
from dqn_agent import Agent
from config import Config
import torch
from CybORG import CybORG
from CybORG.Agents.Wrappers import *
import inspect
from buffers import *
from agents.dqn.attention import TemporalAttn
import os


# Main entry point
if __name__ == "__main__":
    track = False
    config_defaults = {'gamma': 0.9,
                       'epsilon': 1.0,
                       'batch_size': 100,
                       'update_step': 50,
                       'episode_length': 100,
                       'learning_rate': 0.005,
                       'training_length': 4000,
                       'priority': False,
                       'exploring_steps': 100,
                       'rnd': False,
                       'pre_obs_norm': 10,
                       'attention': True}
    config = Config(config_dict=config_defaults)

    show_train_curve = not track

    agent_name = 'Blue'
    #agents = {'Red': B_lineAgent}
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim')
    # wrappers = FixedFlatWrapper(EnumActionWrapper(cyborg))
    # environment = OpenAIGymWrapper(env=wrappers, agent_name='Blue')
    environment = OpenAIGymWrapper(agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(cyborg))))

    print('Environment Initalised...')

    # initalise the state and agent
    action_space = environment.action_space.n
    obs_space = environment.observation_space.shape[0]
    agent = Agent(action_space=action_space, state_space=int(obs_space / 10), gamma=config.gamma,
                  epsilon=config.epsilon, lr=config.learning_rate, batch_size=config.batch_size, model='attention')
    attention = TemporalAttn(obs_space)
    print('Agent Created...')
    observation = environment.reset()

    total_episodes = []
    total_losses = []
    total_rewards = []
    total_ep_lengths = []
    rolling_ep_len_avg = []
    total_successful_episodes = []
    xp_buffer = PriorityReplayBuffer() if config.priority else ReplayBuffer()


    while config.batch_size > len(xp_buffer):
        observation = environment.reset()

        for step in range(0, config.episode_length):
            action = np.random.randint(0, action_space)
            next_obs, reward, done, _ = environment.step(action)
            # get index of top N idx with highest attention

            # sort index by size
            # reduce observation by the idx selected by attention
            xp_buffer.add_transition([observation, action, reward, next_obs, done])

            # observation = next_obs
    print('Training...')

    total_step_number = 0
    num_eps_this_form = 0

    episode = 0
    start_time = time.time()
    while episode < config.training_length:
        ep_loss = []
        episode_rewards = []
        episode_disc_rewards = 0
        observation = environment.reset()
        for step in range(0, config.episode_length):
            minibatch = xp_buffer.sample(config.batch_size)

            action = agent.get_action(observation)

            next_observation, reward, done, infos = environment.step(action)
            total_reward = reward
            loss = agent.dqn.train_q_network(minibatch, rnd=config.rnd, priority=config.priority)
            xp_buffer.add_transition([observation, action, reward, next_observation, done])

            episode_rewards.append(reward)
            observation = next_observation
            ep_loss.append(loss)
        agent.update_epsilon()
        num_eps_this_form += 1

        if 1:
            print("{:<5}{:<6}{:>2}{:<15}{:>.3f}{:<15}{:>.3f}{:<22}{:>.3f} {:<.3f} {:> .3f} {:<.3f}{:<40}".format(
                str(episode),
                'AGENT: ', 1,
                ' EP_LOSS_AV: ', float(np.mean(ep_loss) / (step + 1)) if ep_loss else 0,
                ' EP_REWARD: ', float(sum(episode_rewards)),
                ' REWARD MIN/MAX/MEAN/SD: ',
                float(min([list(episode_rewards)[j] for j in range(len(episode_rewards))])),
                float(max([list(episode_rewards)[j] for j in range(len(episode_rewards))])),
                float(np.mean([float(list(episode_rewards)[j]) for j in range(len(episode_rewards))])),
                float(np.std([list(episode_rewards)[j] for j in range(len(episode_rewards))])),
                ' ACTION: ' + str(infos['action'])
            ))
            total_successful_episodes.append(len(np.where(np.mean(episode_rewards, axis=0) == 0)[0]))
            total_losses.append(sum(ep_loss) / step if ep_loss else 0)
            total_episodes.append(episode)
            total_rewards.append(sum(episode_rewards))
            total_ep_lengths.append(step)
        episode += 1
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Total run time: ')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    environment.close()
    agent.save_model('a2c with rnd')

    if show_train_curve:
        plt.plot(total_episodes, total_rewards, color='orange')
        plt.title('Mean reward of all agents in the episode')
        plt.grid()
        plt.savefig('./saved_models/a2c/reward.png')
        plt.show()

        plt.plot(total_episodes, total_losses)
        plt.title('Mean Value loss of all agents in the episode')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('MSE of Advantage')
        plt.savefig('./saved_models/a2c/loss.png')
        plt.show()

        plt.plot(total_episodes, total_successful_episodes, "o")
        plt.title('Number of episodes defended successfully')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('Number of successful episodes')
        plt.savefig('./saved_models/a2c/success.png')
        plt.show()








