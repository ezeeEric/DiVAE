# -*- coding: utf-8 -*-
"""
Sequential Variational Autoencoder

A architecture using multiple VAEs, where each VAE is conditioned on the
previous VAEs input. This is to create a hierarchical dependency if there is two
or more input images which depend on each other.

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn
from diVAE import VariationalAutoEncoder,AutoEncoder
from DiVAE import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

class SequentialVariationalAutoEncoder(AutoEncoder):

    def __init__(self, **kwargs):
        super(SequentialVariationalAutoEncoder, self).__init__(**kwargs)
        self._type="sVAE"
        self._autoencoders={}
        self._encoder_nodes={}
        self._reparam_nodes={}
        self._decoder_nodes={}
        
        input_enc=0
        input_dec=0

        for i,dim in enumerate(self._input_dimension):
            self._reparam_nodes[i]=(self._config.encoder_hidden_nodes[-1],self._latent_dimensions)

            #define network structure
            self._encoder_nodes[i]=[]
            self._decoder_nodes[i]=[]
            
            #for each new calo layer, add input dimension
            input_enc+=dim
            #for each new calo layer, add input dimension
            input_dec+=self._latent_dimensions if i==0 else self._input_dimension[i-1]

            enc_node_list=[input_enc]+self._config.encoder_hidden_nodes

            for num_nodes in range(0,len(enc_node_list)-1):
                nodepair=(enc_node_list[num_nodes],enc_node_list[num_nodes+1])
                self._encoder_nodes[i].append(nodepair)
            
            dec_node_list=[input_dec]+self._config.decoder_hidden_nodes+[dim]

            for num_nodes in range(0,len(dec_node_list)-1):
                nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
                self._decoder_nodes[i].append(nodepair)
        self.dummy=nn.ModuleList([])
   
    #TODO this is definitely a hack. The VAE submodules in this class are
    #somehow not properly registered. This means no nn.module.parameters are broadcasted
    #to this class, despite each VAE being properly registered.
    def flatten_network_dependency(self):
        for key,vae in self._autoencoders.items():
            self.dummy.extend(self._autoencoders[key].get_modules())
    
    def create_networks(self):
        logger.debug("Creating Network Structures")

        for i,dim in enumerate(self._input_dimension):
            self._autoencoders[i]=VariationalAutoEncoder.init_with_nodelist(dim=dim,
                                                        cfg=self._config,
                                                        actfct=self._activation_fct,
                                                        enc_nodes=self._encoder_nodes[i],
                                                        rep_nodes=self._reparam_nodes[i],
                                                        dec_nodes=self._decoder_nodes[i]
                                                        )
            self._autoencoders[i].create_networks()   
        self.flatten_network_dependency()

    def forward(self, x, label):
        outputs=[]
        mus=[]
        logvars=[]
        for i,dim in enumerate(self._input_dimension):
            current_vae=self._autoencoders[i]
            x_transformed=x[i].view(-1, dim)

            q = current_vae.encoder.encode(x_transformed)
            mu = current_vae._reparam_layers['mu'](q)
            logvar = current_vae._reparam_layers['var'](q)
            zeta = current_vae.reparameterize(mu, logvar)
            x_recon = current_vae.decoder.decode(zeta)
            
            outputs.append(x_recon)
            mus.append(mu)
            logvars.append(logvar)
            print(x_recon.shape)
            exit()
        return outputs, mus, logvars

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
    
    def reparameterize(self, mu, logvar):
        """ 
        Sample epsilon from the normal distributions. Return mu+epsilon*sqrt(var),
        corresponding to random sample from Gaussian with mean mu and variance var.
        """
        
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
                   