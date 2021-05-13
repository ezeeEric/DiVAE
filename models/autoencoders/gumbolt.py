"""
GumBolt Discrete Variational Autoencoder - PyTorch Implementation

Author : Abhi (abhishek@myumanitoba.ca)
"""
# Torch imports
import torch
from torch.nn import BCEWithLogitsLoss

# DiVAE.models imports
from models.autoencoders.dvaepp import DiVAEPP
from models.rbm.rbm import RBM
from models.samplers.pcd import PCD
from models.networks.hierarchicalEncoder import HierarchicalEncoder
from models.networks.basicCoders import BasicDecoder

# DiVAE.utils imports
from utils.dists.distributions import Bernoulli

from DiVAE import logging
logger = logging.getLogger(__name__)

class GumBolt(DiVAEPP):
    
    def __init__(self, **kwargs):
        super(DiVAEPP, self).__init__(**kwargs)
        self._model_type = "GumBolt"
        self._bce_loss = BCEWithLogitsLoss(reduction="none")
        
    def _create_encoder(self):
        """
        - Overrides _create_encoder in dvaepp.py
        
        Returns:
            Hierarchical Encoder instance
        """
        logger.debug("GumBolt::_create_encoder")
        return HierarchicalEncoder(
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            n_encoder_layer_nodes=self.n_encoder_layer_nodes,
            n_encoder_layers=self.n_encoder_layers,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)
    
    def loss(self, input_data, fwd_out):
        """
        - Overrides loss in dvaepp.py
        
        Returns:
            Autoencoding loss + KL divergence loss
        """
        logger.debug("GumBolt::loss")
        
        kl_loss, cross_entropy, pos_energy, neg_energy = self.kl_divergence(fwd_out.post_logits,
                                                                            fwd_out.post_samples)
        
        ae_loss_matrix = -fwd_out.output_distribution.log_prob_per_var(input_data.view(-1, self._flat_input_size))
        ae_loss_per_sample = torch.sum(ae_loss_matrix,1)
        ae_loss = torch.mean(ae_loss_per_sample)
        
        loss = ae_loss + kl_loss
        
        return {"loss":loss, "ae_loss":ae_loss, "kl_loss":kl_loss,
                "cross_entropy":cross_entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}

    def kl_divergence(self, post_logits, post_samples, is_training=True):
        """
        - Compute KLD b.w. hierarchical posterior and RBM prior using GumBolt trick
        - Overrides kl_divergence in dvaepp.py
        - Uses negative energy expectation value as an approximation to logZ
        
        Args:
            post_logits: List of posterior logits (logit_q_z)
            post_samples: List of posterior samples (zeta)
        Returns:
            kl_loss: "Approximate integral KLD" loss whose gradient equals the
                     gradient of the true KLD loss
        """
        logger.debug("GumBolt::kl_divergence")
        
        # Concatenate all hierarchy levels
        logits_q_z = torch.cat(post_logits, 1)
        post_zetas = torch.cat(post_samples, 1)
        
        # Compute cross-entropy b/w post_logits and post_samples
        cross_entropy = - self._bce_loss(logits_q_z, post_zetas)
        cross_entropy = torch.mean(torch.sum(cross_entropy, 1), 0)
        
        # Compute positive energy expval using hierarchical posterior samples
        
        # Number of hidden and visible variables on each side of the RBM
        num_var_rbm = (self.n_latent_hierarchy_lvls * self._latent_dimensions)//2
        
        # Compute positive energy contribution to the KL divergence
        post_zetas_vis, post_zetas_hid = post_zetas[:, :num_var_rbm], post_zetas[:, num_var_rbm:]
        pos_energy = self.energy_exp(post_zetas_vis, post_zetas_hid)
        
        # Compute gradient contribution of the logZ term
        rbm_visible_samples, rbm_hidden_samples = self.sampler.block_gibbs_sampling()
        rbm_vis, rbm_hid = rbm_visible_samples.detach(), rbm_hidden_samples.detach()
        neg_energy = - self.energy_exp(rbm_vis, rbm_hid)
        
        kl_loss = cross_entropy + pos_energy + neg_energy
        return kl_loss, cross_entropy, pos_energy, neg_energy 
        
    def energy_exp(self, rbm_vis, rbm_hid):
        """
        - Compute the energy expectation value
        
        Returns:
            rbm_energy_exp_val : mean(-vis^T W hid - a^T hid - b^T vis)
        """
        logger.debug("GumBolt::energy_exp")
        
        # Broadcast W to (pcd_batchSize * nVis * nHid)
        w, vbias, hbias = self.prior.weights, self.prior.visible_bias, self.prior.hidden_bias
        w = w + torch.zeros((rbm_vis.size(0),) + w.size(), device=rbm_vis.device)
        vbias = vbias.to(rbm_vis.device)
        hbias = hbias.to(rbm_hid.device)
        
        # Prepare H, V for torch.matmul()
        # Change V.size() from (batchSize * nVis) to (batchSize * 1 * nVis)
        vis = rbm_vis.unsqueeze(2).permute(0, 2, 1)
        # Change H.size() from (batchSize * nHid) to (batchSize * nHid * 1)
        hid = rbm_hid.unsqueeze(2)
        
        batch_energy = (- torch.matmul(vis, torch.matmul(w, hid)).reshape(-1) 
                        - torch.matmul(rbm_vis, vbias)
                        - torch.matmul(rbm_hid, hbias))
        
        return torch.mean(batch_energy, 0)