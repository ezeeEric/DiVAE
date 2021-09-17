"""
GumBolt implementation for Calorimeter data
V6 - Changed hierarchical encoder creation
"""

# DiVAE.models imports
from models.autoencoders.gumboltCaloV5 import GumBoltCaloV5
from models.networks.hierarchicalEncoderV2 import HierarchicalEncoderV2

from DiVAE import logging
logger = logging.getLogger(__name__)

class GumBoltCaloV6(GumBoltCaloV5):
    
    def __init__(self, **kwargs):
        super(GumBoltCaloV6, self).__init__(**kwargs)
        self._model_type = "GumBoltCaloV6"
 
    def _create_encoder(self):
        """
        - Overrides _create_encoder in GumBoltCaloV5.py
        
        Returns:
            Hierarchical Encoder instance
        """
        logger.debug("GumBoltCaloV6::_create_encoder")
        return HierarchicalEncoderV2(
            input_dimension=self._flat_input_size+1,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            n_encoder_layer_nodes=self.n_encoder_layer_nodes,
            n_encoder_layers=self.n_encoder_layers,
            skip_latent_layer=False,
            smoother="Gumbel",
            cfg=self._config)