
"""
Conditional Variational Autoencoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
from models.variationalAE import VariationalAutoEncoder

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)

class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):

    def __init__(self, **kwargs):
        super(ConditionalVariationalAutoEncoder, self).__init__(**kwargs)
        self._model_type="cVAE"
        #define network structure
        self._encoder_nodes=[]
        self._decoder_nodes=[]
        
        #TODO hydra: is there a built-in feature for list comprehension?
        enc_hidden_nodes=[int(i) for i in self._config.model.encoder_hidden_nodes.split(",")]

        enc_node_list=[self._flat_input_size+1]+enc_hidden_nodes

        for num_nodes in range(0,len(enc_node_list)-1):
            nodepair=(enc_node_list[num_nodes],enc_node_list[num_nodes+1])
            self._encoder_nodes.append(nodepair)
        
        self._reparam_nodes=(enc_hidden_nodes[-1],self._latent_dimensions)
        
        #TODO hydra: is there a built-in feature for list comprehension?
        dec_hidden_node_list=[int(i) for i in self._config.model.decoder_hidden_nodes.split(",")]

        dec_node_list=[self._latent_dimensions+1]+dec_hidden_node_list+[self._flat_input_size]

        for num_nodes in range(0,len(dec_node_list)-1):
            nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
            self._decoder_nodes.append(nodepair)
        pass

    def forward(self, input_data, label):
        #see definition for explanation
        out=self._output_container.clear()

        input_data_transformed=input_data.view(-1, self._flat_input_size)
        label_unsqueezed=label.unsqueeze(-1)

        input_data_cat=torch.cat([input_data_transformed,label_unsqueezed],dim=1)
        z = self.encoder.encode(input_data_cat)
        out.mu = self._reparam_layers['mu'](z)
        out.logvar = self._reparam_layers['var'](z)
        zeta = self.reparameterize(out.mu, out.logvar)
        out.zeta_cat=torch.cat([zeta,label_unsqueezed],dim=1)
        
        out.output_data = self.decoder.decode(out.zeta_cat)

        return out

    def generate_samples(self):
        outlist=[]
        
        #how many samples to generate per number
        n_samples_per_nr=int(self._config.n_generate_samples)
        
        #TODO hydra list comprehension
        target_nrs=[int(i) for i in self._config.model.target_numbers.split(",")]
        for i in target_nrs:
            rnd_input=torch.randn((n_samples_per_nr,self._reparam_nodes[1]))
            target=torch.full((n_samples_per_nr, 1), i, dtype=torch.float32)
            rnd_input_cat=torch.cat([rnd_input,target], dim=1)
            output = self.decoder.decode(rnd_input_cat)
            output.detach()
            outlist.append(output)
        return torch.cat(outlist)
