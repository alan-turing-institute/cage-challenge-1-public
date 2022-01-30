
from gym import spaces
from agents.hierachy_agents.scaffold_env import *
import ray.rllib.agents.ppo as ppo
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from agents.hierachy_agents.sub_agents import sub_agents

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

        relative_path = os.path.abspath(os.getcwd())[:62] + '/cage-challenge-1'
        #print(relative_path)
        self.BLcheckpoint_pointer = relative_path + sub_agents['B_line_trained']
        self.RMcheckpoint_pointer = relative_path + sub_agents['RedMeander_trained']
        sub_config = {
            "env": CybORGScaffBL,
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
        ModelCatalog.register_custom_model("CybORG_PPO_Model", TorchModel)

        # Restore the checkpointed model
        self.RM_def = ppo.PPOTrainer(config=sub_config, env=CybORGScaffBL)
        sub_config['env'] = CybORGScaffRM
        self.BL_def = ppo.PPOTrainer(config=sub_config, env=CybORGScaffRM)
        self.BL_def.restore(self.BLcheckpoint_pointer)
        self.RM_def.restore(self.RMcheckpoint_pointer)

        self.steps = 1
        self.agent_name = 'BlueHier'

        #action space is 2 for each trained agent to select from
        self.action_space = spaces.Discrete(2)

        # observations for controller is a sliding window of 4 observations
        self.observation_space = spaces.Box(-1.0,1.0,(52*4,), dtype=float)

        #defuault observation is 4 lots of nothing
        self.observation = np.zeros((52*4))

        self.action = None
        self.env = self.BLenv

    # reset doesnt reset the sliding window of the agent so it can differentiate between
    # agents across episode boundaries
    def reset(self):
        self.steps = 1
        #rest the environments of each attacker
        self.BLenv.reset()
        self.RMenv.reset()
        if random.choice([0,1]) == 0:
            self.env = self.BLenv
        else:
            self.env = self.RMenv
        return np.zeros((52*4))

    def step(self, action=None):
        # select agent
        if action == 0:
            # get action from agent trained against the B_lineAgent
            agent_action = self.BL_def.compute_single_action(self.observation[-52:])
        elif action == 1:
            # get action from agent trained against the RedMeanderAgent
            agent_action = self.RM_def.compute_single_action(self.observation[-52:])
        else:
            print('something went terribly wrong, old sport')
        observation, reward, done, info = self.env.step(agent_action)

        # update sliding window
        observation = np.append(self.observation[52:], observation)
        self.observation = observation

        self.steps += 1
        if self.steps == 100:
            return observation, reward, True, info
        assert(self.steps <= 100)
        result = observation, reward, done, info
        return result

    def seed(self, seed=None):
        random.seed(seed)
