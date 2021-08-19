"""
GumBolt implementation for Calorimeter data
V2 - Using 2-headed decoder to recconstruct an activation mask
in addition to the calorimeter depositions

Author : Abhi (abhishek@myumanitoba.ca)
"""

# Torch imports
import torch
from torch.nn import ReLU, MSELoss, BCEWithLogitsLoss, Sigmoid

# DiVAE.models imports
from models.autoencoders.gumbolt import GumBolt
from models.networks.basicCoders import BasicDecoderV2, BasicDecoderV3

from DiVAE import logging
logger = logging.getLogger(__name__)

class GumBoltCaloV2(GumBolt):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloV2, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloV2"
        self._energy_activation_fct = ReLU()
        self._hit_activation_fct = Sigmoid()
        self._output_loss = MSELoss(reduction="none")
        self._hit_loss = BCEWithLogitsLoss(reduction="none")
        
    def forward(self, x):
        """
        - Overrides forward in dvaepp.py
        
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
        
        output_hits, output_activations = self.decoder(post_samples)
        
        out.output_hits = output_hits        
        out.output_activations = self._energy_activation_fct(output_activations) * torch.where(self._hit_activation_fct(output_hits) > 0.5, 1., 0.)
        return out
    
    def loss(self, input_data, fwd_out):
        logger.debug("loss")
        
        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        ae_loss = self._output_loss(input_data, fwd_out.output_activations)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)
        
        hit_loss = self._hit_loss(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.))
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
        loss = ae_loss + kl_loss + hit_loss
        
        return {"loss":loss, "ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
    
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
            
            output_hits, output_activations = self.decoder(prior_samples)
            sample = self._energy_activation_fct(output_activations) * torch.where(self._hit_activation_fct(output_hits) > 0.5, 1., 0.)
            samples.append(sample)
            
        return torch.cat(samples, dim=0)
    
    def _create_decoder(self):
        logger.debug("GumBoltCaloV2:_create_decoder")
        return BasicDecoderV3(node_sequence=self._decoder_nodes, activation_fct=self._activation_fct,  cfg=self._config)