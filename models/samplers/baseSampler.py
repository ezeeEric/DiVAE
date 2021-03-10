"""
Abstract Base Class for Samplers
"""

from torch import nn

class BaseSampler(nn.Module):
    def __init__(self, learning_rate, momentum_coefficient, n_gibbs_sampling_steps, weight_decay_factor, **kwargs):
        super(BaseSampler, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.n_gibbs_sampling_steps = n_gibbs_sampling_steps
        self.weight_decay_factor = weight_decay_factor
    
    def run_training(self):
        raise NotImplementedError

    def sample_from_visible(self):
        raise NotImplementedError

    def sample_from_hidden(self):
        raise NotImplementedError
