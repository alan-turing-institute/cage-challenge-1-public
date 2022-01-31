import os.path as path
from gym import spaces
from agents.hierachy_agents.scaffold_env import *
import ray.rllib.agents.ppo as ppo
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from agents.hierachy_agents.sub_agents import sub_agents

from agents.hierachy_agents.CybORGAgent import CybORGAgent

class TorchModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                 name)
        torch.nn.Module.__init__(self)

        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class HierEnv(gym.Env):
    # Env parameters
    max_steps = 100 # Careful! There are two other envs!
    mem_len = 4

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    """The CybORGAgent env"""

    def __init__(self, config: EnvContext):
        agents = {
            'Red': RedMeanderAgent  # B_lineAgent , #RedMeanderAgent, 'Green': GreenAgent
        }

        self.cyborg = CybORG(self.path, 'sim', agents={'Red':RedMeanderAgent})
        self.RMenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.cyborg = CybORG(self.path, 'sim', agents={'Red':B_lineAgent})
        self.BLenv  = ChallengeWrapper(env=self.cyborg, agent_name='Blue')


        #relative_path = #'cage-challenge-1' #[:62], os.path.abspath(os.getcwd()) +
        #print(relative_path)
        two_up = path.abspath(path.join(__file__, "../../.."))
        self.BL_checkpoint_pointer = two_up + '/log_dir/b_line_trained/PPO_CybORGAgent_e81fb_00000_0_2022-01-29_11-23-39/checkpoint_002500/checkpoint-2500'#relative_path + sub_agents['B_line_trained']
        self.RM_checkpoint_pointer = two_up + '/log_dir/meander_trained/PPO_CybORGAgent_3c456_00000_0_2022-01-27_20-39-34/checkpoint_001882/checkpoint-1882'#relative_path + sub_agents['RedMeander_trained']
    
        
        sub_config_BL = {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                "vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }
            }
        }

        sub_config_M = {
            "env": CybORGAgent,
            "env_config": {
                "null": 0,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "custom_model": "CybORG_PPO_Model",
                #"vf_share_layers": True,
            },
            "lr": 0.0001,
            # "momentum": tune.uniform(0, 1),
            "num_workers": 0,  # parallelism
            "framework": "torch",  # May also use "tf2", "tfe" or "torch" if supported
            "eager_tracing": True,  # In order to reach similar execution speed as with static-graph mode (tf default)
            "vf_loss_coeff": 0.01,  # Scales down the value function loss for better comvergence with PPO
            "in_evaluation": True,
            'explore': False,
            "exploration_config": {
                "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }
            }
        }

        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)

        # Restore the checkpointed model
        self.RM_def = ppo.PPOTrainer(config=sub_config_M, env=CybORGAgent)
        #sub_config['env'] = CybORGAgent
        self.BL_def = ppo.PPOTrainer(config=sub_config_BL, env=CybORGAgent)
        self.BL_def.restore(self.BL_checkpoint_pointer)
        self.RM_def.restore(self.RM_checkpoint_pointer)

        self.steps = 0
        self.agent_name = 'BlueHier'

        #action space is 2 for each trained agent to select from
        self.action_space = spaces.Discrete(2)

        # observations for controller is a sliding window of 4 observations
        self.observation_space = spaces.Box(-1.0,1.0,(self.mem_len,52), dtype=float)

        #defuault observation is 4 lots of nothing
        self.observation = np.zeros((self.mem_len,52))

        self.action = None
        self.env = self.BLenv

    # reset doesnt reset the sliding window of the agent so it can differentiate between
    # agents across episode boundaries
    def reset(self):
        self.steps = 0
        #rest the environments of each attacker
        self.BLenv.reset()
        self.RMenv.reset()
        if random.choice([0,1]) == 0:
            self.env = self.BLenv
        else:
            self.env = self.RMenv
        return np.zeros((self.mem_len,52))

    def step(self, action=None):
        # select agent
        if action == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-1:])
        elif action == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action = self.RM_def.compute_single_action(self.observation[-1:])
        else:
            print('something went terribly wrong, old sport')
        observation, reward, done, info = self.env.step(agent_action)

        # update sliding window
        self.observation = np.roll(self.observation, -1, 0) # Shift left by one to bring the oldest timestep on the rightmost position
        self.observation[self.mem_len-1] = observation      # Replace what's on the rightmost position

        self.steps += 1
        if self.steps == self.max_steps:
            return self.observation, reward, True, info
        assert(self.steps <= self.max_steps)
        result = self.observation, reward, done, info
        return result

    def seed(self, seed=None):
        random.seed(seed)
