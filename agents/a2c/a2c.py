import torch
import numpy as np
from agents.a2c.distributions import Categorical



class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(PolicyNetwork, self).__init__()
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.device = torch.device(dev)
        self.output_space = output_dimension
                                                                            
                   
                                       

        self.network = NeuralNetwork(input_dimension, device=self.device)
        self.dist = Categorical(self.network.output_size, self.output_space, device=self.device)
    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def recurrent_hidden_size(self):
        return self.network.recurrent_hidden_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.network(inputs.to(self.device), rnn_hxs.to(self.device), masks.to(self.device))
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.network(inputs.to(self.device), rnn_hxs.to(self.device), masks.to(self.device))
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.network(inputs.to(self.device), rnn_hxs.to(self.device), masks.to(self.device))
        dist = self.dist(actor_features.to(self.device))

        action_log_probs = dist.log_probs(action.to(self.device))
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class MLPBase(torch.nn.Module):
    def __init__(self, recurrent, rnn_input_size, hidden_size):
        super(MLPBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = torch.nn.GRU(rnn_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0)
                elif 'weight' in name:
                    torch.nn.init.orthogonal_(param)
    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size




class NeuralNetwork(MLPBase):
    def __init__(self, input_dimension, device, recurrent=False, hidden_size=64):
        super(NeuralNetwork, self).__init__(recurrent, input_dimension, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = torch.nn.Sequential(
            init_(torch.nn.Linear(input_dimension, hidden_size)), torch.nn.Tanh(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh()).to(device)

        self.critic = torch.nn.Sequential(
            init_(torch.nn.Linear(input_dimension, hidden_size)), torch.nn.Tanh(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh()).to(device)

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1)).to(device)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

