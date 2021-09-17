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
    def __init__(self, node_sequence=None, activation_fct=None, create_module_list=True, cfg=None, create_network=True, **kwargs):
        super(Network, self).__init__(**kwargs)
        
        self._config=cfg
        
        self._layers=nn.ModuleList([]) if create_module_list else None
        self._node_sequence=node_sequence
        self._activation_fct=activation_fct

        if self._node_sequence and create_module_list and create_network:
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
    
# Two headed network - forks at the beginning
class NetworkV2(Network):
    def __init__(self, create_module_list=True, create_network=True, **kwargs):
        super(NetworkV2, self).__init__(create_network=False, **kwargs)
        print("Initializing NetworkV2")
        self._layers2 = nn.ModuleList([]) if create_module_list else None
        
        if self._node_sequence and create_module_list and create_network:
            self._create_network()
        
    def _create_network(self):
        super(NetworkV2, self)._create_network()
        print("Node sequence : ", self._node_sequence)
        for node in self._node_sequence:
            self._layers2.append(
                nn.Linear(node[0], node[1])
            )
        return
    
# Two headed network - forks at the last 2 layers
class NetworkV3(Network):
    def __init__(self, create_module_list=True, create_network=True, **kwargs):
        super(NetworkV3, self).__init__(create_network=False, **kwargs)
        print("Initializing NetworkV3")
        
        self._layers2 = nn.ModuleList([]) if create_module_list else None
        self._layers3 = nn.ModuleList([]) if create_module_list else None
        
        if self._node_sequence and create_module_list and create_network:
            self._create_network()
            
    def _create_network(self):
        n_layers = len(self._node_sequence)
        for idx, node in enumerate(self._node_sequence):
            if idx < n_layers-2:
                self._layers.append(nn.Linear(node[0],node[1]))
            else:
                self._layers2.append(nn.Linear(node[0],node[1]))
                self._layers3.append(nn.Linear(node[0],node[1]))
        return
    
if __name__=="__main__":
    logger.debug("Testing Networks")

    logger.debug("Success")