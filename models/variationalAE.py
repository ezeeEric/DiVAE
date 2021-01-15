# -*- coding: utf-8 -*-
"""
Vanilla Variational Autoencoder Model

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch
from torch import nn
from models.autoencoder import AutoEncoder

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

# Vanilla Variational Autoencoder implementation
# Adds VAE specific reparameterisation, loss and forward call to AutoEncoder framework
class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self._model_type="VAE"

        #define network structure
        self._encoder_nodes=[]
        self._decoder_nodes=[]
        
        enc_node_list=[self._input_dimension]+self._config.encoder_hidden_nodes

        for num_nodes in range(0,len(enc_node_list)-1):
            nodepair=(enc_node_list[num_nodes],enc_node_list[num_nodes+1])
            self._encoder_nodes.append(nodepair)
        
        self._reparam_nodes=(self._config.encoder_hidden_nodes[-1],self._latent_dimensions)
        
        dec_node_list=[self._latent_dimensions]+self._config.decoder_hidden_nodes+[self._input_dimension]

        for num_nodes in range(0,len(dec_node_list)-1):
            nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
            self._decoder_nodes.append(nodepair)
    
    #Factory method to create VAEs with set encoder nodes
    @classmethod
    def init_with_nodelist(cls,dim,cfg,actfct, enc_nodes, rep_nodes, dec_nodes):
        assert enc_nodes is not None and rep_nodes is not None and dec_nodes is not None,\
            "Need defined nodelist for this type of initialisation"
        vae=cls(input_dimension=dim,config=cfg,activation_fct=actfct)              
        vae._encoder_nodes=enc_nodes
        vae._reparam_nodes=rep_nodes
        vae._decoder_nodes=dec_nodes
        return vae

    def get_modules(self):
        return self.encoder,self._reparam_layers,self.decoder

    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        
        self._reparam_layers=nn.ModuleDict(
            {'mu':  nn.Linear(self._reparam_nodes[0],self._reparam_nodes[1]),
             'var': nn.Linear(self._reparam_nodes[0],self._reparam_nodes[1])
             })
        
        self.decoder=self._create_decoder()
        return

    def reparameterize(self, mu, logvar):
        """ 
        Sample epsilon from the normal distributions. Return mu+epsilon*sqrt(var),
        corresponding to random sample from Gaussian with mean mu and variance var.
        """
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
    
    def generate_samples(self,n_samples=100):
        """ 
        Similar to fwd. only skip encoding part...
        """
        # Draw a rnd var z~N[0,1] and feed it through the decoder
        rnd_input=torch.randn((n_samples,self._reparam_nodes[1]))
        zeta=rnd_input 
        output = self.decoder.decode(zeta)
        return output

    def loss(self, x, x_recon, mu, logvar):
        logger.debug("VAE Loss")
        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(x_recon, x.view(-1, self._input_dimension), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
                            
    def forward(self, x):
        z = self.encoder.encode(x.view(-1, self._input_dimension))
        mu = self._reparam_layers['mu'](z)
        logvar = self._reparam_layers['var'](z)
        zeta = self.reparameterize(mu, logvar)
        x_recon = self.decoder.decode(zeta)
        return x_recon, mu, logvar, zeta

if __name__=="__main__":
    logger.info("Running variationalautoencoder.py directly") 
    model=VariationalAutoEncoder()
    print(model)
    logger.info("Success")