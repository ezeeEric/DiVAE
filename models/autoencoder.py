
"""
Vanilla Autoencoder Model

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
from torch import nn
from models.autoencoderbase import AutoEncoderBase

from utils.networks import BasicEncoder,BasicDecoder

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

# Vanilla Autoencoder implementation
class AutoEncoder(AutoEncoderBase):

    def __init__(self, **kwargs):
        super(AutoEncoder,self).__init__(**kwargs)
        self._model_type="AE"

        #define network structure
        self._encoder_nodes=[]
        self._decoder_nodes=[]
        
        enc_node_list=[self._input_dimension]+self._config.encoder_hidden_nodes+[self._latent_dimensions]

        for num_nodes in range(0,len(enc_node_list)-1):
            nodepair=(enc_node_list[num_nodes],enc_node_list[num_nodes+1])
            self._encoder_nodes.append(nodepair)
       
        dec_node_list=[self._latent_dimensions]+self._config.decoder_hidden_nodes+[self._input_dimension]

        for num_nodes in range(0,len(dec_node_list)-1):
            nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
            self._decoder_nodes.append(nodepair)

        #only works if input_data, output_data in [0,1]
        self._loss_fct= nn.functional.binary_cross_entropy

    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
        return

    def _create_encoder(self):
        logger.debug("_create_encoder")
        return BasicEncoder(node_sequence=self._encoder_nodes, activation_fct=self._activation_fct)

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return BasicDecoder(node_sequence=self._decoder_nodes, activation_fct=self._activation_fct, output_activation_fct=nn.Sigmoid())

    def forward(self, input_data):
        #see definition for einput_dataplanation
        out=self._output_container.clear()
        
        out.zeta = self.encoder.encode(input_data.view(-1,self._input_dimension))
        out.output_data = self.decoder.decode(zeta)

        return out
    
    def loss(self, input_data, fwd_out):
        loss=self._loss_fct(fwd_out.output_data, input_data.view(-1,self._input_dimension), reduction='sum')
        return loss

if __name__=="__main__":
    logger.info("Running autoencoder.py directly") 
    model=AutoEncoder()
    print(model)
    logger.info("Success")