# -*- coding: utf-8 -*-
"""
Conditional Variational Autoencoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
from utils.diVAE import VariationalAutoEncoder

class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):

    def __init__(self, **kwargs):
        super(ConditionalVariationalAutoEncoder, self).__init__(**kwargs)
        self._model_type="cVAE"
        #define network structure
        self._encoder_nodes=[]
        self._decoder_nodes=[]
        
        enc_node_list=[self._input_dimension+1]+self._config.encoder_hidden_nodes

        for num_nodes in range(0,len(enc_node_list)-1):
            nodepair=(enc_node_list[num_nodes],enc_node_list[num_nodes+1])
            self._encoder_nodes.append(nodepair)
        
        self._reparam_nodes=(self._config.encoder_hidden_nodes[-1],self._latent_dimensions)
        
        dec_node_list=[self._latent_dimensions+1]+self._config.decoder_hidden_nodes+[self._input_dimension]

        for num_nodes in range(0,len(dec_node_list)-1):
            nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
            self._decoder_nodes.append(nodepair)
        pass

    def forward(self, x, label):
        x_transformed=x.view(-1, self._input_dimension)
        label_unsqueezed=label.unsqueeze(-1)
        x_cat=torch.cat([x_transformed,label_unsqueezed],dim=1)
        z = self.encoder.encode(x_cat)
        mu = self._reparam_layers['mu'](z)
        logvar = self._reparam_layers['var'](z)
        zeta = self.reparameterize(mu, logvar)
        zeta_cat=torch.cat([zeta,label_unsqueezed],dim=1)
        x_recon = self.decoder.decode(zeta_cat)
        return x_recon, mu, logvar, zeta_cat

    def generate_samples(self,n_samples_per_nr=5, nrs=[0,1,2]):
        """ 
        Similar to fwd. only skip encoding part...
        """
        outlist=[]
        for i in nrs:
            rnd_input=torch.randn((n_samples_per_nr,self._reparam_nodes[1]))
            target=torch.full((n_samples_per_nr, 1), i, dtype=torch.float32)
            rnd_input_cat=torch.cat([rnd_input,target], dim=1)
            output = self.decoder.decode(rnd_input_cat)
            outlist.append(output)
        return torch.cat(outlist)