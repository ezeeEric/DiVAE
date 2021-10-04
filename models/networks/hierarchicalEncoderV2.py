"""
Hierarchical Encoder V2

Changes the way the encoder is constructed wrt V1. The hierarchy levels >1 are
using the same structure as the first level.

"""
import torch.nn as nn  

from models.networks.hierarchicalEncoder import HierarchicalEncoder

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)

class HierarchicalEncoderV2(HierarchicalEncoder):
    def __init__(self, **kwargs):
        super(HierarchicalEncoderV2, self).__init__(**kwargs)
        
    def _create_hierarchy_network(self,level=0, skip_latent_layer=False):
        layers=[self.num_input_nodes+(level*self.n_latent_nodes)]+list(self._config.model.encoder_hidden_nodes)+[self.n_latent_nodes]
        moduleLayers=nn.ModuleList([])
        for l in range(len(layers)-1):
            moduleLayers.append(nn.Linear(layers[l], layers[l+1]))
            #apply the activation function for all layers except the last (latent) layer 
            act_fct = nn.Identity() if l==len(layers)-2 else self.activation_fct
            moduleLayers.append(act_fct)

        sequential=nn.Sequential(*moduleLayers)
        return sequential
    
if __name__=="__main__":
    logger.debug("Testing Networks")
    logger.debug("Success")