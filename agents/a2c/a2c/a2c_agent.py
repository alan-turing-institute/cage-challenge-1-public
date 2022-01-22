import torch
import os
from agents.a2c.a2c.a2c import PolicyNetwork
from agents.a2c.a2c.rnd.rnd import RNDAgentA2c
from agents.a2c.a2c.rollout import RolloutStorage

class A2CAgent:
    def __init__(self, args):
        self.step               = 0
        self.actor_critic       = PolicyNetwork(args.obs_space, args.action_space)
        self.val_loss_coef      = args.val_loss_coef
        self.entropy_coef       = args.entropy_coef
        self.max_grad_norm      = args.max_grad_norm
        if args.rnd == True:
            self.rnd            = RNDAgentA2c(input_size=args.obs_space, output_size=args.action_space)
            self.update_prop    = args.update_prop
        else:
            self.rnd            = None
        self.optimiser          = torch.optim.RMSprop(self.actor_critic.parameters(), args.learning_rate, eps=args.epsilon, alpha=args.alpha)
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.rollouts = RolloutStorage(steps=args.episode_length, processes=args.processes, output_dimensions=args.action_space,
                                  input_dimensions=args.obs_space)
        self.device = torch.device(dev)

    def update(self):
        obs_shape = self.rollouts.observations.size()[2:]
        action_shape = self.rollouts.actions.size()[-1]
        num_steps, num_processes, _ = self.rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            self.rollouts.observations[:-1].view(-1, *obs_shape),
            self.rollouts.masks[:-1].view(-1, 1),
            self.rollouts.actions.view(-1, action_shape))
        if self.rnd:
            predict_state_feature, target_state_feature = self.rnd.forward(self.rollouts.observations[:-1].squeeze(-1).squeeze(-1))
            predict_state_feature = predict_state_feature.reshape(predict_state_feature.shape[0]*predict_state_feature.shape[1]*predict_state_feature.shape[2], 1)
            target_state_feature = target_state_feature.reshape(target_state_feature.shape[0]*target_state_feature.shape[1]*target_state_feature.shape[2], 1)
            forward_loss = torch.nn.MSELoss()(predict_state_feature.squeeze(-1), target_state_feature.detach().squeeze(-1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.rollouts.returns[:-1].to(self.device) - values.to(self.device)
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimiser.zero_grad()
        if self.rnd:
            torch.autograd.set_detect_anomaly(True)
            (value_loss * self.val_loss_coef + action_loss -
                dist_entropy * self.entropy_coef + forward_loss).backward()
            self.rnd.update(self.rollouts.observations[:-1].squeeze(-1).squeeze(-1))
        else:
            (value_loss * self.val_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()

        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimiser.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def save_model(self):
        path = os.path.abspath(os.getcwd())
        if not os.path.exists(path + '/saved_models'):
            os.mkdir(path +'/saved_models')
        if not os.path.exists(path +'/saved_models/a2c'):
            os.mkdir(path +'/saved_models/a2c')
        model_to_save = {'agent':'agent', 'ac_state_dict': self.actor_critic.state_dict(),
                         'opt_state_dict':self.optimiser.state_dict()}
        if self.rnd:
            model_to_save['rnd_state_dict'] = self.rnd.rnd_predictor.state_dict()
        torch.save(model_to_save, path + '/saved_models/a2c/actor_critic.pt')

    def load(self, file_path):
        check_point = torch.load(file_path)

        self.actor_critic.load_state_dict(check_point['ac_state_dict'])
        self.optimiser.load_state_dict(check_point['opt_state_dict'])
        if self.rnd:
            self.rnd.rnd_predictor.load_state_dict(check_point['rnd_state_dict'])
        self.actor_critic.train()
        if self.rnd:
            self.rnd.rnd_predictor.train()
        print('Loaded model: '+str(check_point['agent']))

    def get_action(self, observation, action_space):
        with torch.no_grad():
            value, continuous_action, action_log_prob = self.actor_critic.act(
                self.rollouts.observations[self.step])

        masks = torch.FloatTensor([[0.0]])
        bad_masks = torch.FloatTensor([[0.0]])
        self.rollouts.insert(observation, continuous_action, action_log_prob, value, 0, masks, bad_masks)

        self.rollouts.after_update()




