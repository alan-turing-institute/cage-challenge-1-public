import time
from matplotlib import pyplot as plt
import numpy as np
from agents.a2c.env_utils import make_envs_as_vec
from agents.a2c.a2c.a2c_agent import A2CAgent
from agents.a2c.a2c.rollout import RolloutStorage
from agents.a2c.a2c.rnd.rnd import RunningMeanStd
import torch
from CybORG import CybORG
from CybORG.Agents import *
import inspect
from config import configure


# Main entry point
if __name__ == "__main__":
    track = False

    show_train_curve = not track

    agent_name = 'Blue'
    agents = {'Red': B_lineAgent}
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim', agents=agents)
    processes = 4
    environments = make_envs_as_vec(seed=0, processes=processes, gamma=0.95, env=cyborg)

    print('Environment Initalised...')

    #initalise the state and agent
    args = configure()
    action_space = environments.action_space.n
    obs_space    = environments.observation_space.shape[0]
    agent = A2CAgent(args)
    print('Agent Created...')
    observation = environments.reset()
    agent.step = 0
    r_mean = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(processes, obs_space))

    total_episodes = []
    total_losses = []
    total_rewards = []
    total_ep_lengths = []
    rolling_ep_len_avg = []
    total_successful_episodes = []
    print('Initialise observation standardisation...')
    for ep in range(args.pre_obs_norm):
        obs = environments.reset()
        for step in range(args.episode_length):
            actions = np.array([np.random.randint(0, action_space) for i in range(processes)])
            obs, _, _, _ = environments.step(actions)

            obs_rms.update(obs)
    print('Finish observation standardisation')



    print('Training...')

    total_step_number = 0
    num_eps_this_form = 0

    episode = 0
    start_time = time.time()
    while any(succ_eps != 4 for succ_eps in total_successful_episodes[:50]) or len(total_successful_episodes) < 50:
        ep_loss = []
        episode_rewards = []
        episode_disc_rewards = 0
        observations = environments.reset()
        agent.step = 0
        for step in range(0, args.episode_length):
            with torch.no_grad():
                value, continuous_action, action_log_prob = agent.actor_critic.act(agent.rollouts.observations[step])

            observation, reward, done, infos = environments.step(continuous_action)
            if args.rnd:
                intrinsic_reward = agent.rnd.compute_intrinsic_reward((observation - obs_rms.mean) / np.sqrt(obs_rms.var)).detach().numpy()
                mean, std, count = np.mean(intrinsic_reward), np.std(intrinsic_reward), len(intrinsic_reward)
                r_mean.update_from_moments(mean, std ** 2, count)
                intrinsic_reward /= np.sqrt(r_mean.var)
                intrinsic_reward = np.clip(intrinsic_reward, -0.1, 0.1)
                obs_rms.update(observation)
                total_reward = reward + intrinsic_reward
            else:
                total_reward = reward
            episode_rewards.append(reward)


            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor( [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            agent.rollouts.insert(observation, continuous_action, action_log_prob, value, total_reward, masks, bad_masks)
        with torch.no_grad():
            next_value = agent.actor_critic.get_value(agent.rollouts.observations[-1]).detach()
        agent.rollouts.compute_returns(next_value, gamma=0.99)

        value_loss, action_loss, dist_entropy = agent.update()
        agent.rollouts.after_update()
        ep_loss.append(value_loss)
        num_eps_this_form += 1


        if 1:
            for i in range(len(reward)):

                print("{:<5}{:<6}{:>2}{:<15}{:>.3f}{:<15}{:>.3f}{:<22}{:>.3f} {:<.3f} {:> .3f} {:<.3f}{:<40}".format(
                    str(episode),
                    'AGENT: ', i+1,
                    ' EP_LOSS_AV: ', float(np.mean(ep_loss)/(step+1)) if ep_loss else 0,
                    ' EP_REWARD: ', float(sum(sum(episode_rewards))/processes),
                    ' REWARD MIN/MAX/MEAN/SD: ', float(min([list(episode_rewards)[j][i] for j in range(len(episode_rewards))])),
                    float(max([list(episode_rewards)[j][i] for j in range(len(episode_rewards))])), float(np.mean([float(list(episode_rewards)[j][i]) for j in range(len(episode_rewards))])),
                    float( np.std([list(episode_rewards)[j][i] for j in range(len(episode_rewards))])),
                               ' ACTION: ' + str(infos[i]['action'])
                ))
            print('\n')
            total_successful_episodes.append(processes - len(np.where(np.mean(episode_rewards, axis=0) < 0)[0]))
            total_losses.append(sum(ep_loss)/step if ep_loss else 0)
            total_episodes.append(episode)
            total_rewards.append(sum(sum(episode_rewards))/processes)
            total_ep_lengths.append(step)
        episode += 1
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Total run time: ')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    environments.close()
    agent.save_model('a2c')

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








