import torch
import numpy as np

class RNDNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension):
        #
        super(RNDNeuralNetwork, self).__init__()
        # Define the network layers of the RND network
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=124)
        self.layer_2 = torch.nn.Linear(in_features=124, out_features=124)
        self.output_layer = torch.nn.Linear(in_features=124, out_features=output_dimension)

        #for p in self.modules():
        #    if isinstance(p, torch.nn.Linear):
        #        p.bias.data.zero_()
        #        torch.nn.init.orthogonal_(p.weight, np.sqrt(2))

        #torch.nn.init.orthogonal_(self.output_layer.weight, 0.01)
        #self.output_layer.bias.data.zero_()

    def forward(self, input):
        if type(input) != torch.Tensor:
            input = torch.tensor(input, dtype=torch.float32)
        layer_1_output  = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output  = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output          = self.output_layer(layer_2_output)
        return output

class RNDAgentA2c():
    def __init__(self, input_size=1, output_size=24):
        self.rnd_target     = RNDNeuralNetwork(input_dimension=input_size,output_dimension=output_size)
        self.rnd_predictor  = RNDNeuralNetwork(input_dimension=input_size,output_dimension=output_size)
        self.opt            = torch.optim.Adam(self.rnd_predictor.parameters(), lr = 0.0001)

    def forward(self, input):
        target_out  = self.rnd_target.forward(input)
        predict_out = self.rnd_predictor.forward(input)
        return predict_out, target_out

    def compute_intrinsic_reward(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        target_output       = self.rnd_target(observation).detach()
        prediction          = self.rnd_predictor(observation)
        reward              = torch.pow(prediction - target_output, 2).sum(1)
        return reward

    def update(self, observation):
        output = self.compute_intrinsic_reward(observation)
        self.opt.zero_grad()
        output.sum().backward()
        self.opt.step()



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

