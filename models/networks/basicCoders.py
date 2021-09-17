"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn

from models.networks.networks import Network, NetworkV2, NetworkV3

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)

class BasicEncoder(Network):
    def __init__(self,**kwargs):
        super(BasicEncoder, self).__init__(**kwargs)

    def forward(self, x):
        logger.debug("Encoder::encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x

class BasicDecoder(Network):
    def __init__(self,output_activation_fct=nn.Identity(),**kwargs):
        super(BasicDecoder, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct

    def forward(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1 and self._output_activation_fct:
                x=self._output_activation_fct(layer(x))
            else:
                x=self._activation_fct(layer(x))
        return x
    

class BasicDecoderV2(NetworkV2):
    def __init__(self, output_activation_fct=nn.Identity(),**kwargs):
        super(BasicDecoderV2, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct

    def forward(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        x1, x2 = x, x
        for idx, (layer1, layer2) in enumerate(zip(self._layers, self._layers2)):
            if idx==nr_layers-1 and self._output_activation_fct:
                x1 = self._output_activation_fct(layer1(x1))
                x2 = self._output_activation_fct(layer2(x2))
            else:
                x1 = self._activation_fct(layer1(x1))
                x2 = self._activation_fct(layer2(x2))
        return x1, x2
    
class BasicDecoderV3(NetworkV3):
    def __init__(self, output_activation_fct=nn.Identity(), **kwargs):
        super(BasicDecoderV3, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct
        
    def forward(self, x):
        logger.debug("Decoder::decode")
        
        for layer in self._layers:
            x=self._activation_fct(layer(x))
            
        nr_layers=len(self._layers2)
        x1, x2 = x, x
        
        for idx, (layer2, layer3) in enumerate(zip(self._layers2, self._layers3)):
            if idx==nr_layers-1 and self._output_activation_fct:
                x1 = self._output_activation_fct(layer2(x1))
                x2 = self._output_activation_fct(layer3(x2))
            else:
                x1 = self._activation_fct(layer2(x1))
                x2 = self._activation_fct(layer3(x2))
        return x1, x2
    
if __name__=="__main__":
    logger.debug("Testing Networks")

    logger.debug("Success")