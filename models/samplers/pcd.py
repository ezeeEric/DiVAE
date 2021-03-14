"""
PCD Sampler

Author: Abhi (abhishek@myumanitoba.ca)
"""

from models.samplers.baseSampler import BaseSampler
import torch

class PCD(BaseSampler):
    
    def __init__(self, batchSize, RBM, **kwargs):
        super(PCD, self).__init__(**kwargs)
        
        self._RBM = RBM
        self._MCState = (torch.rand(batchSize, self._RBM.get_visible_bias().size(0)) >= 
                         torch.rand(batchSize, self._RBM.get_visible_bias().size(0))).float()
        
    def hidden_samples(self, visible_states):
        """
        Sample batch of hidden states given a batch of visible states
        
        Args:
            visible_states : Tensor, Dims=(batchSize * nVisibleNodes)
        Output:
            hidden_states sample : Tensor, Dims=(batchSize * nHiddenNodes)
        """
        hidden_activations = (torch.matmul(visible_states, self._RBM.get_weights())
                          + self._RBM.get_hidden_bias())
        hidden_probs = torch.sigmoid(hidden_activations)
        return (hidden_probs >= torch.rand(hidden_probs.size())).float()
        
    def visible_samples(self, hidden_states):
        """
        Sample batch of visible states given a batch of hidden states
        
        Args:
            hidden_states : Tensor, Dims=(batchSize * nHiddenNodes)
        Output:
            visible_states sample : Tensor, Dims=(batchSize * nVisibleNodes)
        """
        visible_activations = (torch.matmul(hidden_states, self._RBM.get_weights().t()) 
                           + self._RBM.get_visible_bias())
        visible_probs = torch.sigmoid(visible_activations)
        return (visible_probs >= torch.rand(visible_probs.size())).float()
    
    def block_gibbs_sampling(self):
        """
        Block Gibbs sampling with initialization through a persistent Markov Chain
        
        Returns:
            visible_states : Batch of visible states at end of Gibbs sampling, Dims=(batchSize * nVisibleNodes)
            hidden_states : Batch of hidden states at end of Gibbs sampling, Dims=(batchSize * nHiddenNodes)
        """
        visible_states = self._MCState
        for step in range(self.n_gibbs_sampling_steps):
            hidden_states = self.hidden_samples(visible_states)
            visible_states = self.visible_samples(hidden_states)
        
        self._MCState = visible_states
        
        return visible_states, hidden_states