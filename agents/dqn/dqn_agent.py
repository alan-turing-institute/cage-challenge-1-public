from bs4 import BeautifulSoup
import difflib
import re
from dqn import DQN, DuelingDQN, AverageDQN
import torch
import os
import numpy as np

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, state_space, action_space, epsilon = 1.0, gamma=0.9,
                 lr=0.0001, model='dqn', batch_size=32):
        # set the number of steps after which to update the target network
        # set epsilon
        self.epsilon = epsilon
        # set gamma
        self.gamma = gamma
        # set reward values

        # initalise the current state
        self.state = None

        self.num_actions = action_space

        # initalise reward for episodes
        self.total_reward = None
        # curiosity using the RND model

        if model == 'dqn':
            self.model_type = 'dqn'
            # initalise the Q-Network
            self.dqn = DQN(gamma=self.gamma, lr=lr, batch_size=batch_size, input_dimension=state_space, output_dimension=action_space)
        elif model == 'attention':
            self.dqn = DQN(gamma=self.gamma, lr=lr, batch_size=batch_size, attention=True, input_dimension=state_space, output_dimension=action_space)
        elif model == 'dueling':
            self.model_type = 'dueling_dqn'
            self.dqn = DuelingDQN(batch_size=batch_size,lr=lr)
        elif model == 'average':
            self.model_type = 'average_dqn'
            self.dqn = AverageDQN(batch_size=batch_size, lr=lr)
        # login to the webapp and save the sessions for requests and Selenium


        # testing params
        self.episode_rewards = []
        self.q_vals ={}



    def save_model(self):
        path = os.path.abspath(os.getcwd())
        if not os.path.exists(path + '/saved_models'):
            os.mkdir(path +'/saved_models')
        if not os.path.exists(path +'/saved_models/' + self.model_type):
            os.mkdir(path +'/saved_models/' + self.model_type)
        model_to_save = {'dqn_q_net_state_dict': self.dqn.q_network.state_dict(),
                         'dqn_target_state_dict': self.dqn.target_q_values.state_dict(),
                         'opt_state_dict':self.dqn.optimiser.state_dict()}
        if self.rnd:
            model_to_save['rnd_state_dict'] = self.rnd.rnd_predictor.state_dict()
        torch.save(model_to_save, path + '/saved_models/' + self.model_type + '/dqn.pt')



    def update_network(self):
        self.dqn.update_target_network()


    def update_network(self):
        self.dqn.update_target_network()


    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= 0.999
        else:
            self.epsilon = 0.1
        return

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(range(0, self.num_actions))
        else:
            state_q_values = self.dqn.predict_q_values(state)
            action = np.argmax(state_q_values)
        return action



