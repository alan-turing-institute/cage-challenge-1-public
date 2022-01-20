import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage:
    def __init__(self, steps=10, input_dimensions=2, output_dimensions=27,
                 rnn_state_size=1, processes=2):

        self.observations       = torch.zeros(steps + 1, processes, input_dimensions)
        self.rewards            = torch.zeros(steps, processes, 1)
        self.value_preds        = torch.zeros(steps, processes, 1)
        self.action_log_probs   = torch.zeros(steps, processes, 1)
        self.actions            = torch.zeros(steps, processes, 1).long()
        self.returns             = torch.zeros(steps + 1, processes, 1)
        self.masks              = torch.ones(steps + 1, processes, 1)
        self.bad_masks          = torch.ones(steps + 1, processes, 1)
        self.num_steps          = steps
        self.step               = 0


    def insert(self, observation, actions, action_log_probs, value_preds, rewards, masks, bad_masks):
        self.observations[self.step + 1].copy_(torch.tensor(observation))
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        try:
            self.rewards[self.step].copy_(torch.tensor(rewards).unsqueeze(-1))
        except:
            self.rewards[self.step].copy_(torch.tensor(rewards))
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])


    def compute_returns(self, next_val, gamma, use_time_limit=True):
        if use_time_limit:
            self.returns[-1] = next_val
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (self.returns[step + 1] * \
                                      gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                     + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            self.returns[-1] = next_val
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generateor(self, advs, num_mini_batch=32, minibatch_size=None):
        steps, processes = self.rewards.size()[0:2]
        batch_size = processes * steps

        if minibatch_size is None:
            minibatch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True)

        for indices in sampler:
            observation_batch           = self.observations[:-1].view(-1, *self.observations.size()[2:])[indices]
            actions_batch               = self.actions.view(-1, self.actions.size(-1))[indices]
            val_pred_batch              = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch                = self.returns[:-1].view(-1, 1)[indices]
            masks_batch                 = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_prob_batch   = self.action_log_probs.view(-1, 1,)[indices]

            if advs is None:
                adv_target = None
            else:
                adv_target = advs.view(-1, 1)[indices]

            yield observation_batch, rnn_state_batch, actions_batch, \
                  val_pred_batch, return_batch, masks_batch, old_action_log_prob_batch, adv_target
