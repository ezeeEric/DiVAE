"""
DVAE++ implementation for Calorimeter data

Author : Abhi (abhishek@myumanitoba.ca)
"""

# Torch imports
import torch
from torch.nn import ReLU, MSELoss

# DiVAE.models imports
from models.autoencoders.dvaepp import DiVAEPP

from DiVAE import logging
logger = logging.getLogger(__name__)

class DiVAEPPCalo(DiVAEPP):
    
    def __init__(self, **kwargs):
        super(DiVAEPPCalo, self).__init__(**kwargs)
        self._model_type = "DiVAEPPCalo"
        self._output_activation_fct = ReLU()
        self._output_loss = MSELoss(reduction="none")
        
    def forward(self, x):
        """
        - Overrides forward in discreteVAE.py
        
        Returns:
            out: output container 
        """
        logger.debug("forward")
        
        #see definition for explanation
        out=self._output_container.clear()
      	
	    #TODO data prep - study if this does good things
        input_data_centered=x.view(-1, self._flat_input_size)#-self._dataset_mean
        
	    #Step 1: Feed data through encoder
        out.beta, out.post_logits, out.post_samples = self.encoder(input_data_centered)
        post_samples = torch.cat(out.post_samples, 1)
        
        output_activations = self.decoder(post_samples)
        #out.output_activations = torch.clamp(output_activations+self._train_bias, min=-88., max=88.)
        out.output_activations = self._output_activation_fct(output_activations)
        return out
    
    def loss(self, input_data, fwd_out):
        logger.debug("loss")
        
        kl_loss, cross_entropy, entropy, neg_energy=self.kl_divergence(fwd_out.beta, fwd_out.post_logits, fwd_out.post_samples)
        ae_loss = self._output_loss(input_data, fwd_out.output_activations)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)
        
        loss = ae_loss + kl_loss
        
        return {"loss":loss, "ae_loss":ae_loss, "kl_loss":kl_loss,
                "cross_entropy":cross_entropy, "entropy":entropy, "neg_energy":neg_energy}
    
    def generate_samples(self, num_samples=64):
        """
        generate_samples()
        """
        num_iterations = max(num_samples//self.sampler.get_batch_size(), 1)
        samples = []
        for i in range(num_iterations):
            rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling()
            rbm_vis = rbm_visible_samples.detach()
            rbm_hid = rbm_hidden_samples.detach()
            prior_samples = torch.cat([rbm_vis, rbm_hid], 1)
            samples.append(self._output_activation_fct(self.decoder(prior_samples)))
            
        return torch.cat(samples, dim=0)