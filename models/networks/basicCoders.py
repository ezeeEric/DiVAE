
"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn

from models.networks.networks import Network

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)

#Implements encode()
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

#Implements decode()
class BasicDecoder(Network):
    def __init__(self,output_activation_fct=None,**kwargs):
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

if __name__=="__main__":
    logger.debug("Testing Networks")

    logger.debug("Success")