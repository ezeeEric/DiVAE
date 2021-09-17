"""
GumBolt implementation for Calorimeter data
V3 - Energy conditioning added over V2

Author : Abhi (abhishek@myumanitoba.ca)
"""

# Torch imports
import torch
from torch.nn import ReLU, MSELoss, BCEWithLogitsLoss, L1Loss, Sigmoid

# DiVAE.models imports
from models.autoencoders.gumbolt import GumBolt
from models.networks.basicCoders import BasicDecoderV2, BasicDecoderV3

# DiVAE.utils imports
from utils.dists.gumbelmod import GumbelMod

from DiVAE import logging
logger = logging.getLogger(__name__)

class GumBoltCaloV3(GumBolt):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloV3, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloV3"
        self._energy_activation_fct = ReLU()
        self._hit_activation_fct = Sigmoid()
        self._output_loss = MSELoss(reduction="none")
        self._hit_loss = BCEWithLogitsLoss(reduction="none")
        
        self._hit_smoothing_dist_mod = GumbelMod()
        
    def forward(self, x, is_training):
        """
        - Overrides forward in dvaepp.py
        
        Returns:
            out: output container 
        """
        logger.debug("forward")
        
        #see definition for explanation
        out=self._output_container.clear()
        
	    #Step 1: Feed data through encoder
        out.beta, out.post_logits, out.post_samples = self.encoder(x, is_training)
        post_samples = torch.cat(out.post_samples, 1)
        
        output_hits, output_activations = self.decoder(post_samples)
        
        out.output_hits = output_hits
        beta = torch.tensor(self._config.model.beta_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        out.output_activations = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, is_training)
        return out
    
    def loss(self, input_data, fwd_out):
        logger.debug("loss")
        
        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits, fwd_out.post_samples)
        ae_loss = self._output_loss(input_data, fwd_out.output_activations)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0)
        
        hit_loss = self._hit_loss(fwd_out.output_hits, torch.where(input_data > 0, 1., 0.))
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
        
        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
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
            beta = torch.tensor(self._config.model.beta_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
            sample = self._energy_activation_fct(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta, False)
            samples.append(sample)
            
        return torch.cat(samples, dim=0)
    
    def _create_decoder(self):
        logger.debug("GumBoltCaloV3:_create_decoder")
        return BasicDecoderV3(node_sequence=self._decoder_nodes, activation_fct=self._activation_fct,  cfg=self._config)