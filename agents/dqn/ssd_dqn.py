import torch

class StreamLayer(torch.nn.Module):
    def __init__(self, input_dimension=128, output_dimension=1):
        super(StreamLayer, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=output_dimension)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        return output

# The NeuralNetwork class inherits the torch.nn.Module class, which represents a neural network.
class NeuralNetwork(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, output_dimension, layer_seps):
        # Call the initialisation function of the parent class.
        super(NeuralNetwork, self).__init__()
        # Define the network layers.
        layers = []
        for seperation in layer_seps:
            layers.append(torch.nn.Linear(in_features=seperation, out_features=128))

        self.sep_1_layer_1 = torch.nn.Linear(in_features=layer_seps[0], out_features=256)
        self.sep_1_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_2_layer_1 = torch.nn.Linear(in_features=layer_seps[1], out_features=256)
        self.sep_2_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_3_layer_1 = torch.nn.Linear(in_features=layer_seps[2], out_features=256)
        self.sep_3_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_4_layer_1 = torch.nn.Linear(in_features=layer_seps[3], out_features=256)
        self.sep_4_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.sep_5_layer_1 = torch.nn.Linear(in_features=layer_seps[4], out_features=256)
        self.sep_5_layer_2 = torch.nn.Linear(in_features=256, out_features=128)

        self.comb_layer_1 = torch.nn.Linear(in_features=128*5, out_features=128)
        #torch.nn.init.xavier_normal_(self.layer_1.weight)
        self.comb_layer_2 = torch.nn.Linear(in_features=128, out_features=96)

        self.output_layer = torch.nn.Linear(in_features=96, out_features=output_dimension)
        #torch.nn.init.xavier_normal_(self.output_layer.weight)


    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, seps):
        sep_1_out = torch.tanh(self.sep_1_layer_2(torch.tanh(self.sep_1_layer_1(seps[0]))))
        sep_2_out = torch.tanh(self.sep_2_layer_2(torch.tanh(self.sep_2_layer_1(seps[1]))))
        sep_3_out = torch.tanh(self.sep_3_layer_2(torch.tanh(self.sep_3_layer_1(seps[2]))))
        sep_4_out = torch.tanh(self.sep_4_layer_2(torch.tanh(self.sep_4_layer_1(seps[3]))))
        sep_5_out = torch.tanh(self.sep_5_layer_2(torch.tanh(self.sep_5_layer_1(seps[4]))))

        comb_1_out = torch.tanh(self.comb_layer_1(torch.cat((sep_1_out,sep_2_out,sep_3_out,sep_4_out,sep_5_out), 1)))
        comb_2_out = torch.tanh(self.comb_layer_2(comb_1_out))

        output = self.output_layer(comb_2_out)

        return output



# The DQN class
class DQN:

    # class initialisation function.
    #output = 42500 for the sliced vector action space
    def __init__(self, layer_partitions, input_slices, output_dimension=24, gamma=0.9, lr=0.0001,
                 batch_size=32, rnd_params=None):
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.device = torch.device(dev)
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = NeuralNetwork(output_dimension=output_dimension, layer_seps=layer_partitions).to(self.device)
        self.target_network = NeuralNetwork(output_dimension=output_dimension, layer_seps=layer_partitions).to(self.device)
        self.update_target_network()
        self.input_slices = input_slices
        # optimiser used when updating the Q-network.
        # learning rate determines how big each gradient step is during backpropagation.
        params = self.q_network.parameters()

        self.optimiser = torch.optim.Adam(params, lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size



    # Function to train the Q-network
    def train_q_network(self, minibatch, priority):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch, priority)
        if priority:
            updated_priorities = loss + 1e-5
            loss = loss.mean()
            rnd_loss = None

        q_loss = loss
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        if priority:
            return (loss.item(), q_loss.item()), updated_priorities
        else:
            return loss.item()

    # Function to calculate the loss for a minibatch.
    def _calculate_loss(self, minibatch, priority):
        if priority:
            states, actions, rewards, next_states, buffer_indices, weights, dones = minibatch
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            states, actions, rewards, next_states, dones = minibatch
        state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.int32).to(self.device)
        # Calculate the predicted q-values for the current state
        seps_state_tensor = []
        seps_next_state_tensor = []
        for slice in self.input_slices:
            if len(slice) == 2:
                seps_next_state_tensor.append(next_state_tensor[:, slice[0]:slice[1]])
                seps_state_tensor.append(state_tensor[:, slice[0]:slice[1]])
            elif len(slice) == 1:
                seps_state_tensor.append(state_tensor[:, slice[0]].unsqueeze(-1))
                seps_next_state_tensor.append(next_state_tensor[:, slice[0]].unsqueeze(-1))
        state_q_values = self.q_network.forward(seps_state_tensor)
        state_action_q_values = state_q_values.gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
        # Get the q-values for then next state
        next_state_q_values = self.target_network.forward(seps_next_state_tensor).detach()  # Use .detach(), so that the target network is not updated
        # Get the maximum q-value
        next_state_max_q_values = next_state_q_values.max(1)[0]
        # Calculate the target q values
        target_state_action_q_values = reward_tensor + self.gamma * next_state_max_q_values * (1 - done_tensor)
        # Calculate the loss between the current estimates, and the target Q-values
        loss = torch.nn.MSELoss()(state_action_q_values, target_state_action_q_values)
        if priority:
            loss = loss * weight_tensor
        # Return the loss
        del state_tensor

        return loss

    def predict_q_values(self, state):
        if type(state) != torch.Tensor:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.clone().detach().unsqueeze(0).type(torch.float32).to(self.device)
        seps_state_tensor = []
        for slice in self.input_slices:
            if len(slice) == 2:
                seps_state_tensor.append(state_tensor[:, slice[0]:slice[1]])
            elif len(slice) == 1:
                seps_state_tensor.append(state_tensor[:, slice[0]].unsqueeze(-1))
        predicted_q_value_tensor = self.q_network.forward(seps_state_tensor)
        return predicted_q_value_tensor.data.cpu().numpy()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


