
"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch.nn as nn

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)

#Base Class
class Network(nn.Module):
    def __init__(self, node_sequence=None, activation_fct=None, create_module_list=True, cfg=None, **kwargs):
        super(Network, self).__init__(**kwargs)
        
        self._config=cfg
        
        self._layers=nn.ModuleList([]) if create_module_list else None
        self._node_sequence=node_sequence
        self._activation_fct=activation_fct

        if self._node_sequence and create_module_list:
            self._create_network()
        
    def forward(self, x):
        raise NotImplementedError
    
    def _create_network(self):        
        for node in self._node_sequence:
            self._layers.append(
                nn.Linear(node[0],node[1])
            )
        return

    def get_activation_fct(self):        
        return "{0}".format(self._activation_fct).replace("()","")

if __name__=="__main__":
    logger.debug("Testing Networks")

    logger.debug("Success")