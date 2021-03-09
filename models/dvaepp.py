"""
DVAE++ PyTorch

Author: Abhi (abhishek@myumanitoba.ca)
"""

# PyTorch imports
import torch

# DiVAE imports
from models.autoencoders.discreteVAE import DiVAE
from models.priors.rbm import RBM

from utils.distributions import Bernoulli

from networks.hiEncoder import HierarchicalEncoder
from networks.basicDecoder import BasicDecoder


class DiVAEPP(DiVAE):
    
    def __init__(self, **kwargs):
        super(DiVAE, self).__init__(**kwargs)
        self._model_type = "DiVAEPP"
        
    def _create_encoder(self):
        """
        - Overrides _create_encoder in discreteVAE.py
        
        Returns:
            Hierarchical Encoder instance
        """
        logger.debug("ERROR _create_encoder dummy implementation")
        return HierarchicalEncoder(
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            n_encoder_layer_nodes=self.n_encoder_layer_nodes,
            n_encoder_layers=self.n_encoder_layers,
            skip_latent_layer=False,
            smoother="MixtureExp",
            cfg=self._config)
        
    def kl_divergence(self, post_dists, post_samples, is_training=True):
        """
        - Compute KLD b.w. hierarchical posterior and RBM prior for DVAE++
        - See (https://github.com/QuadrantAI/dvae/blob/master/rbm.py#L139) for original 
          tf implementation
        - Overrides kl_divergence in discreteVAE.py
        Args:
            post_dists: List of dist objects
            post_samples: Approximate post samples (zeta)
        Returns:
            kl_loss: "Approximate integral KLD" loss whose gradient equals the
            gradient of the true KLD loss
        """
        logger.debug("kl_divergence")
        
        # Entropy component of the KLD loss
        entropy = 0
        logit_q = []
        log_ratio = []
        
        # Compute and sum the entropy for each hierarchy level
        for factorial, samples in zip(post_dists, post_samples):
            entropy += torch.sum(factorial.entropy(), 1)
            logit_q.append(factorial.logits)
            log_ratio.append(factorial.log_ratio(samples))
            
        logit_q = torch.cat(logit_q, 1)
        log_ratio = torch.cat(log_ratio, 1)
        samples = torch.cat(post_samples, 1)
        
        num_latent_layers = len(post_dists)
        
        # Mean-field solution for num_latent_layers == 1
        if num_latent_layers == 1:
            log_ratio *= 0.
            
        if is_training:
            cross_entropy = self.cross_entropy_from_hierarchical(logit_q, log_ratio)
        else:
            cross_entropy = - self.log_prob(samples, is_training)
            
        # Add contribution of the logZ term to the cross entropy
        

        kl_loss = cross_entropy - entropy
        
        return kl_loss
    
    def cross_entropy_from_hierarchical(self, logits, log_ratio):
        """
        Compute the cross-entropy b/w a hierarchical distribution and
        an unnormalized Boltzmann machines
        -
        Args:
            logits: Bernoulli logits for all hierarchy levels (Concatenated)
            log_ratio: Log PDF ratio for the smoothing transformations
            
        Returns:
            cross_entropy
        """
        num_var_rbm = (self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions)//2
        
        logit_q1 = logit_q[:, :num_var_rbm]
        logit_q2 = logit_q[:, num_var_rbm:]
        
        log_ratio_1 = log_ratio[:, :num_var_rbm]
        
        q1 = torch.sigmoid(logit_q1)       
        q2 = torch.sigmoid(logit_q2)
        q1_pert = torch.sigmoid(logit_q1 + log_ratio1)
        
        cross_entropy = (-torch.matmul(q1, self.prior.get_visible_bias())-torch.matmul(q2, self.b2) 
                         -torch.sum(torch.matmul(q1_pert, self.w) * q2, 1, keep_dims=True))
        
        cross_entropy = torch.squeeze(cross_entropy, 1)
        return cross_entropy
        
        
        
        
        
        
        
            
        
            
        
            
        
        
        
        
        
        
        
        
        
        
    
        
        
    
            
        
        
            
        
            
        
    
    


