import torch
from a2c.a2c import PolicyNetwork
import os
from a2c.rnd.rnd import RNDAgentA2c

class AgentA2C:
    def __init__(self, action_space, input_space=1, val_loss_coef=0.5,
                 entropy_coef=0.01, lr=0.0001, epsilon=0.001, max_grad_norm=0.5, alpha=0.99,
                 rnd=False, update_prop=0.25, processes=1):
        self.actor_critic       = PolicyNetwork(input_space, action_space)
        self.val_loss_coef      = val_loss_coef
        self.entropy_coef       = entropy_coef
        self.max_grad_norm      = max_grad_norm
        if rnd:
            self.rnd            = RNDAgentA2c(input_size=input_space, output_size=action_space)
            self.update_prop    = update_prop
        else:
            self.rnd            = None
        self.optimiser          = torch.optim.RMSprop(self.actor_critic.parameters(), lr, eps=epsilon, alpha=alpha)
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.device = torch.device(dev)

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.rnn_states[0].view(-1, 1),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        if self.rnd:
            predict_state_feature, target_state_feature = self.rnd.forward(rollouts.observations[:-1].squeeze(-1).squeeze(-1))
            predict_state_feature = predict_state_feature.reshape(predict_state_feature.shape[0]*predict_state_feature.shape[1]*predict_state_feature.shape[2], 1)
            target_state_feature = target_state_feature.reshape(target_state_feature.shape[0]*target_state_feature.shape[1]*target_state_feature.shape[2], 1)
            forward_loss = torch.nn.MSELoss()(predict_state_feature.squeeze(-1), target_state_feature.detach().squeeze(-1))



        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1].to(self.device) - values.to(self.device)
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimiser.zero_grad()
        if self.rnd:
            torch.autograd.set_detect_anomaly(True)
            (value_loss * self.val_loss_coef + action_loss -
                dist_entropy * self.entropy_coef + forward_loss).backward()
            self.rnd.update(rollouts.observations[:-1].squeeze(-1).squeeze(-1))
        else:
            (value_loss * self.val_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()

        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimiser.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def save_model(self, agent):
        path = os.path.abspath(os.getcwd())
        if not os.path.exists(path + '/saved_models'):
            os.mkdir(path +'/saved_models')
        if not os.path.exists(path +'/saved_models/a2c'):
            os.mkdir(path +'/saved_models/a2c')
        model_to_save = {'agent':agent, 'ac_state_dict': self.actor_critic.state_dict(),
                         'opt_state_dict':self.optimiser.state_dict()}
        if self.rnd:
            model_to_save['rnd_state_dict'] = self.rnd.rnd_predictor.state_dict()
        torch.save(model_to_save, path + '/saved_models/a2c/actor_critic.pt')

    def load_model(self):
        path = os.path.abspath(os.getcwd())
        check_point = torch.load(path +'/saved_models/a2c/actor_critic.pt')
        self.actor_critic.load_state_dict(check_point['ac_state_dict'])
        self.optimiser.load_state_dict(check_point['opt_state_dict'])
        if self.rnd:
            self.rnd.rnd_predictor.load_state_dict(check_point['rnd_state_dict'])
        self.actor_critic.train()
        if self.rnd:
            self.rnd.rnd_predictor.train()
        print('Loaded model: '+str(check_point['agent']))
