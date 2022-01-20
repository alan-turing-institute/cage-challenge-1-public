import torch
import numpy as np
from agents.a2c.a2c.distributions import Categorical



class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, slices):
        super(PolicyNetwork, self).__init__()
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.device = torch.device(dev)
        self.input_slices = slices
        self.network = NeuralNetwork(input_dimension, device=self.device, slices=slices)
        self.output_space = output_dimension
        self.dist = Categorical(self.network.output_size, self.output_space, device=self.device)
        dev = 'cpu'
        self.device = torch.device(dev)

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def recurrent_hidden_size(self):
        return self.network.recurrent_hidden_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        #inputs = self.return_as_slices(inputs)
        value, actor_features = self.network(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _= self.network(inputs)
        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.network(inputs.to(self.device))
        dist = self.dist(actor_features.to(self.device))

        action_log_probs = dist.log_probs(action.to(self.device))
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


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



# The NeuralNetwork class inherits the torch.nn.Module class, which represents a neural network.
class SSDMLP(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, layer_seps, device, output_dimension=64):
        # Call the initialisation function of the parent class.
        super(SSDMLP, self).__init__()
        # Define the network layers.

        self.sep_1_layer_1 = torch.nn.Linear(in_features=layer_seps[0], out_features=512).to(device)
        #self.sep_1_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_2_layer_1 = torch.nn.Linear(in_features=layer_seps[1], out_features=512).to(device)
        #self.sep_2_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_3_layer_1 = torch.nn.Linear(in_features=layer_seps[2], out_features=512).to(device)
        #self.sep_3_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_4_layer_1 = torch.nn.Linear(in_features=layer_seps[3], out_features=512).to(device)
        #self.sep_4_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_5_layer_1 = torch.nn.Linear(in_features=layer_seps[4], out_features=512).to(device)
        #self.sep_5_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.comb_layer_1 = torch.nn.Linear(in_features=512*5, out_features=256).to(device)
        #torch.nn.init.xavier_normal_(self.layer_1.weight)
        #self.comb_layer_2 = torch.nn.Linear(in_features=128, out_features=96)

        self.output_layer = torch.nn.Linear(in_features=256, out_features=output_dimension).to(device)
        #torch.nn.init.xavier_normal_(self.output_layer.weight)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, seps):
        sep_1_out = torch.tanh(self.sep_1_layer_1(seps[0]))
        sep_2_out = torch.tanh(self.sep_2_layer_1(seps[1]))
        sep_3_out = torch.tanh(self.sep_3_layer_1(seps[2]))
        sep_4_out = torch.tanh(self.sep_4_layer_1(seps[3]))
        sep_5_out = torch.tanh(self.sep_5_layer_1(seps[4]))

        comb_1_out = torch.tanh(self.comb_layer_1(torch.cat((sep_1_out,sep_2_out,sep_3_out,sep_4_out,sep_5_out), 1)))

        output = self.output_layer(comb_1_out)

        return output



class NeuralNetwork(MLPBase):
    def __init__(self, layer_seps, device, slices, recurrent=False, hidden_size=64):
        super(NeuralNetwork, self).__init__(recurrent, layer_seps, hidden_size)
        if recurrent:
            num_inputs = hidden_size

        self.input_slices = slices
        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = SSDMLP(layer_seps=layer_seps, output_dimension=hidden_size, device=device)


        self.critic = SSDMLP(layer_seps=layer_seps, output_dimension=hidden_size, device=device)

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        self.train()

    def return_as_slices(self, input):
        seps_state_tensor = []
        for slice in self.input_slices:
            if len(slice) == 2:
                seps_state_tensor.append(input[:, slice[0]:slice[1]])
            elif len(slice) == 1:
                seps_state_tensor.append(input[:, slice[0]].unsqueeze(-1))
        return seps_state_tensor

    def forward(self, inputs):
        x = self.return_as_slices(inputs)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


