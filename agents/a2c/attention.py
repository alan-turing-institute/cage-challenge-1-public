import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.device = torch.device(dev)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False).to(self.device)

    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        hidden_states_tensor=torch.tensor(hidden_states, dtype=torch.float32).to(self.device)
        score_first_part = self.fc1(hidden_states_tensor)
        # (batch_size, hidden_size)
        #h_t = hidden_states.unsqueeze(0)
        h_t = hidden_states_tensor
        # (batch_size, time_steps)
        score = torch.mm(score_first_part.T, h_t)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.mm(h_t, attention_weights)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights
