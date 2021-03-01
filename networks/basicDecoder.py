"""
Basic decoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
from networks.network import Network

#Implements decode()
class BasicDecoder(Network):
    def __init__(self,output_activation_fct=None,**kwargs):
        super(BasicDecoder, self).__init__(**kwargs)
        self._output_activation_fct=output_activation_fct

    def decode(self, x):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1 and self._output_activation_fct:
                x=self._output_activation_fct(layer(x))
            else:
                x=self._activation_fct(layer(x))
        return x