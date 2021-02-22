
"""
Sparse Autoencoder Model

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch 

from models.autoencoder import AutoEncoder

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

# Vanilla Autoencoder implementation
class SparseAutoEncoder(AutoEncoder):

    def __init__(self, **kwargs):
        super(SparseAutoEncoder,self).__init__(**kwargs)
        self._model_type="AE"
        self.regularisation_weight=self._config.l1_regularisation_weight
    
    def l1_norm(self, inputs):
        l1_norm = 0
        current_in = inputs
        
        #this assumes encoder and decoder activation fct are the same
        act_fct=self.encoder._activation_fct

        for i, layer in enumerate(self.encoder._layers):
            current_in = act_fct(layer(current_in))
            l1_norm += torch.mean(torch.abs(current_in))
        for i, layer in enumerate(self.decoder._layers):
            current_in = act_fct(layer(current_in))
            l1_norm += torch.mean(torch.abs(current_in))
        return l1_norm

    def loss(self, input_data, fwd_out):
        
        total_loss=0
        l1_regularisation=0
        
        reconstruction_loss=self._loss_fct(fwd_out.output_data, input_data.view(-1,self._flat_input_size), reduction='sum')
        l1_regularisation=self.l1_norm(input_data.view(-1,self._flat_input_size))
        
        total_loss=reconstruction_loss+self.regularisation_weight*l1_regularisation
        
        return total_loss

if __name__=="__main__":
    logger.info("Running autoencoder.py directly") 
    model=AutoEncoder()
    print(model)
    logger.info("Success")