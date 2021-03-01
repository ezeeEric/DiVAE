"""
Simple decoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
from networks.network import Network

class SimpleDecoder(Network):
    def __init__(self, output_nodes=None, output_activation_fct=None, **kwargs):
        super(SimpleDecoder, self).__init__(**kwargs) 
        #last output layer treated separately, as it needs sigmoid activation        
        self._output_activation_fct=output_activation_fct

    def decode(self, z):
        logger.debug("Decoder::decode")
        nr_layers=len(self._layers)
        x_prime=None
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1:
                if self._output_activation_fct:
                    x_prime=self._output_activation_fct(layer(z))
                else:
                    x_prime=self._activation_fct(layer(z))
            else:
                z=self._activation_fct(layer(z))
        return x_prime

    def decode_posterior_sample(self, zeta):
        logger.debug("Decoder::decode")  
        nr_layers=len(self._layers)
        for idx,layer in enumerate(self._layers):
            if idx==nr_layers-1:
                x_prime=self._output_activation_fct(layer(zeta))
            else:
                zeta=self._activation_fct(layer(zeta))
        return x_prime