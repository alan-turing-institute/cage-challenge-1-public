

class Config:
    def __init__(self, config_dict=None):
        self.gamma                      = config_dict['gamma']
        self.epsilon                    = config_dict['epsilon']
        self.batch_size                 = config_dict['batch_size']
        self.update_step                = config_dict['update_step']
        self.episode_length             = config_dict['episode_length']
        self.learning_rate              = config_dict['learning_rate']
        self.training_length            = config_dict['training_length']
        self.priority                   = config_dict['priority']
        self.rnd                        = config_dict['rnd']
        self.pre_obs_norm               = config_dict['pre_obs_norm']

