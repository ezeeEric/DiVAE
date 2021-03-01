"""
Decoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
from torch import nn
from networks.basicDecoder import BasicDecoder

class Decoder(BasicDecoder):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__(**kwargs) 
        self._network=self._create_network()

    def _create_network(self):
        layers=self._node_sequence
        moduleLayers=nn.ModuleList([])
        
        for l in range(len(layers)):
            n_in_nodes=layers[l][0]
            n_out_nodes=layers[l][1]

            moduleLayers.append(nn.Linear(n_in_nodes,n_out_nodes))
            #apply the activation function for all layers except the last
            #(latent) layer 
            act_fct= self._output_activation_fct if l==len(layers)-1 else self._activation_fct
            moduleLayers.append(act_fct)

        sequential=nn.Sequential(*moduleLayers)
        return sequential

    def decode(self, posterior_sample):
        logger.debug("Decoder::decode")
        return self._network(posterior_sample)