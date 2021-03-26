"""
Abstract Base Class for Samplers
"""

from torch import nn

class BaseSampler(nn.Module):
    def __init__(self, n_gibbs_sampling_steps, **kwargs):
        super(BaseSampler, self).__init__(**kwargs)
        self.n_gibbs_sampling_steps = n_gibbs_sampling_steps
    
    def __repr__(self):
        outstring=""
        for key,val in self.__dict__.items():
            outstring+="{0}: {1}\n".format(key,val)
        return outstring
    
    def visible_samples(self):
        raise NotImplementedError

    def hidden_samples(self):
        raise NotImplementedError

    def get_samples(self):
        return NotImplementedError
