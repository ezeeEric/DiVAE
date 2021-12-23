"""
GumBolt implementation for Calorimeter data
CRBM - Change the prior and sampler to Chimera RBM
"""

# DiVAE.models imports
from models.autoencoders.gumboltCaloV6 import GumBoltCaloV6
from models.networks.hierarchicalEncoderV2 import HierarchicalEncoderV2

from models.rbm.chimeraRBM import ChimeraRBM
from models.samplers.pcd import PCD

from DiVAE import logging
logger = logging.getLogger(__name__)

class GumBoltCaloCRBM(GumBoltCaloV6):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloCRBM, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloCRBM"
        
    def _create_prior(self):
        """
        - Override _create_prior in discreteVAE.py
        """
        logger.debug("GumBoltCaloCRBM::_create_prior")
        num_rbm_nodes_per_layer=self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions//2
        return ChimeraRBM(n_visible=num_rbm_nodes_per_layer, n_hidden=num_rbm_nodes_per_layer)
 
    def _create_sampler(self, rbm=None):
        """
        - Overrides _create_sampler in discreteVAE.py
        
        Returns:
            PCD Sampler
        """
        logger.debug("GumBoltCaloCRBM::_create_sampler")
        return PCD(batch_size=self._config.engine.rbm_batch_size, RBM=self.prior, n_gibbs_sampling_steps=self._config.engine.n_gibbs_sampling_steps)