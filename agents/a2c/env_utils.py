from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalise_
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecEnvWrapper)
import torch
import gym
import numpy as np
import sys
sys.path.append("../../cage-challenge-1/CybORG")
from CybORG import CybORG
from CybORG.Agents.Wrappers import *

def list_to_tensor(list):
    return torch.stack(list)

def make_env(rank, environ, seed=0):
    def _thunk():
        environment = environ
        #OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(environment))))
        wrappers = EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(environment)))
        env = OpenAIGymWrapper(env=wrappers, agent_name='Blue')
        env.seed(seed+rank)
        return env
    return _thunk()


def make_envs_as_vec(seed, processes, gamma, env):
    if processes > 1:
        envs = SubprocVecEnv([lambda: make_env(rank=i, environ=env, seed=seed) for i in range(processes)],
                             start_method='spawn')
    else:
        envs = DummyVecEnv([lambda: make_env(rank=0, environ=env,seed=seed)])


    #if len(envs.observation_space.shape) == 1:
    #    envs = VecNormalise(envs, gamma=gamma)
    if processes > 1:
        envs = VecPyTorch(envs)
    else:
        envs = VecPyTorchSingle(envs)

    return envs

class StepLimitMask(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv):
        super(VecPyTorch, self).__init__(venv)

    def reset(self):
        observation = self.venv.reset()

        return observation

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        #actions = actions.numpy()
        try:
            self.venv.step_async(actions)
        except RuntimeError as e:
            self.venv.step_async(actions)

    def step_wait(self):
        observations, reward, done, info = self.venv.step_wait()

        #reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return observations, reward, done, info


class VecBasePyTorch(VecEnvWrapper):
    def __init__(self, venv):
        super(VecBasePyTorch, self).__init__(venv)
    def reset(self):
        observation = self.venv.reset()
        #observation = observation[0]
        return observation

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        #actions = actions.numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        observations, reward, done, info = self.venv.step_wait()
        #observations = observations[0]
        #reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return observations, reward, done, info




class VecPyTorchSingle(VecEnvWrapper):
    def __init__(self, venv):
        super(VecPyTorchSingle, self).__init__(venv)

    def reset(self, agent):
        observation = self.venv.reset(agent=agent)
        return observation

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, reward, done, info = self.venv.step_wait()
        #reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return observations, reward, done, info

class VecNormalise(VecNormalise_):
    def __init__(self, *args, **kwargs):
        super(VecNormalise, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

def extract_obvs_from_wrapper(wrapped_table):
    pass