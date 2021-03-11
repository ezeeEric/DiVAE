"""
Abstract Base Class for Samplers
"""

from torch import nn

class BaseSampler(nn.Module):
    def __init__(self, n_gibbs_sampling_steps, **kwargs):
        super(BaseSampler, self).__init__(**kwargs)
        self.n_gibbs_sampling_steps = n_gibbs_sampling_steps
    
    def run_training(self):
        raise NotImplementedError

    def sample_from_visible(self):
        raise NotImplementedError

    def sample_from_hidden(self):
        raise NotImplementedError
