"""
CDn Gibbs Sampler

Implements Gibbs Sampling/CDn procedure as described in UTML TR 2010–003.
This is an approximation to the original Contrastive Divergence algorithm.
"""

import torch
from torch import nn

from models.samplers.baseSampler import BaseSampler
from DiVAE import logging
logger = logging.getLogger(__name__)

class GibbsSampler(BaseSampler):
    def __init__(self, RBM, **kwargs):
        super(GibbsSampler, self).__init__(**kwargs)
        self._RBM = RBM
        
    def hidden_samples(self, probabilities_visible):
        output_hidden = torch.matmul(probabilities_visible, self._RBM.weights) + self._RBM.hidden_bias
        probabilities_hidden = torch.sigmoid(output_hidden)
        return probabilities_hidden

    def visible_samples(self, probabilities_hidden):
        output_visible = torch.matmul(probabilities_hidden, self._RBM.weights.t()) + self._RBM.visible_bias
        probabilities_visible = torch.sigmoid(output_visible)
        return probabilities_visible

    # Heart of the CDn training: Alternating Gibbs Sampling
    def gibbs_sampling(self,input_sample, n_gibbs_sampling_steps):
        # feed data to hidden layer and sample response
        left=input_sample
        for gibbs_step in range(0,n_gibbs_sampling_steps):
            right=self.hidden_samples(left)
            left=self.visible_samples(right)
        return left, right

        # for step in range(self.n_gibbs_sampling_steps):
        # 	probabilities_visible = self.visible_samples(output_hidden)
        # 	probabilities_hidden = self.hidden_samples(probabilities_visible)
        # 	#When using CDn, only the final update of the hidden nodes should use the probability.
        # 	output_hidden = (probabilities_hidden >= torch.rand(self.n_hidden)).float()
        # return probabilities_visible,probabilities_hidden

    #this method is used to generate samples 
    def get_samples(self, approx_post_samples=[], n_latent_nodes=100, n_latent_hierarchy_lvls=4, n_gibbs_sampling_steps=10):
        logger.debug("get_samples")
        ##Sampling mode: gibbs 
        # start with random numbers left side rbm. Gibbs sampling from
        # trained RBM.
        assert n_latent_hierarchy_lvls%2==0, "Number of hierarchy layers should be even"
        #these are the starting samples for the gibbs sampling - the left
        #(visible) side of the rbm. If no sample is given, random numbers are
        #picked.
        if len(approx_post_samples)<1:
            for i in range(0,n_latent_hierarchy_lvls//2):
                #TODO this range is empirically working... but samples should be
                #binary.
                approx_post_samples.append(-166*torch.rand([n_latent_nodes])+88)
            approx_post_samples=torch.cat(approx_post_samples).detach()
        left,right=self.gibbs_sampling(approx_post_samples,n_gibbs_sampling_steps)
        return [left,right]
