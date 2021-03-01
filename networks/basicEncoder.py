"""
Basic encoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
from networks.network import Network

#Implements encode()
class BasicEncoder(Network):
    def __init__(self,**kwargs):
        super(BasicEncoder, self).__init__(**kwargs)

    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            if self._activation_fct:
                x=self._activation_fct(layer(x))
            else:
                x=layer(x)
        return x