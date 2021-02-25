
"""
Sequential Variational Autoencoder

A architecture using multiple VAEs, where each VAE is conditioned on the
previous VAEs input. This is to create a hierarchical dependency if there is two
or more input images which depend on each other.

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn
from models.autoencoder import AutoEncoder
from models.variationalAE import VariationalAutoEncoder

from utils.helpers import OutputContainer

from DiVAE import logging
logger = logging.getLogger(__name__)

class SequentialVariationalAutoEncoder(AutoEncoder):

    def __init__(self, **kwargs):
        super(SequentialVariationalAutoEncoder, self).__init__(**kwargs)
        self._model_type="sVAE"
        self._autoencoders={}
        self._encoder_nodes={}
        self._reparam_nodes={}
        self._decoder_nodes={}
        
        #TODO this stores the keyword args so they can be used in the
        #init_with_nodelist() call in create_networks. Find better solution.
        #(Refactoring artifact)
        self._kwargs=kwargs
        
        input_enc=0
        input_dec=0

        # input_sizes=self._flat_input_size if isinstance(self._flat_input_size,list) else [self._flat_input_size]

        for i,dim in enumerate(self._flat_input_size):
            self._reparam_nodes[i]=(self._config.encoder_hidden_nodes[-1],self._latent_dimensions)

            #define network structure
            self._encoder_nodes[i]=[]
            self._decoder_nodes[i]=[]
            
            #for each new calo layer, add input dimension
            input_enc+=dim
            #for each new calo layer, add input dimension
            input_dec+=self._latent_dimensions if i==0 else self._flat_input_size[i-1]

            enc_node_list=[input_enc]+self._config.encoder_hidden_nodes

            for num_nodes in range(0,len(enc_node_list)-1):
                nodepair=(enc_node_list[num_nodes],enc_node_list[num_nodes+1])
                self._encoder_nodes[i].append(nodepair)
            
            dec_node_list=[input_dec]+self._config.model.decoder_hidden_nodes+[dim]

            for num_nodes in range(0,len(dec_node_list)-1):
                nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
                self._decoder_nodes[i].append(nodepair)
        self.dummy=nn.ModuleList([])

    #TODO this is an outrageous hack. The VAE submodules in this class are
    #somehow not properly registered. This means no nn.module.parameters are broadcasted
    #from this class, despite each VAE being properly registered. Something
    #Something pytorch. The solution here is to have a dummy variable
    #registering all parameters "flat", i.e. without hierarchy.
    def flatten_network_dependency(self):
        for key,vae in self._autoencoders.items():
            self.dummy.extend(self._autoencoders[key].get_modules())
    
    def create_networks(self):
        logger.debug("Creating Network Structures")

        for i,dim in enumerate(self._flat_input_size):
            self._autoencoders[i]=VariationalAutoEncoder.init_with_nodelist(dim=dim,
                                                        cfg=self._config,
                                                        actfct=self._activation_fct,
                                                        enc_nodes=self._encoder_nodes[i],
                                                        rep_nodes=self._reparam_nodes[i],
                                                        dec_nodes=self._decoder_nodes[i],
                                                        **self._kwargs
                                                        )
            self._autoencoders[i].create_networks()   
        self.flatten_network_dependency()

    def forward(self, input_data):
        #see definition for einput_dataplanation
        out=self._output_container.clear()

        out.outputs=[]
        out.mus=[]
        out.logvars=[]

        for i,dim in enumerate(self._flat_input_size):
            current_vae=self._autoencoders[i]
            #every input is concatenation of previous inputs
            # input_data_transformed=input_data[i].view(-1, dim) if i==0 else torch.cat([input_data_transformed,input_data[i].view(-1, dim)],dim=-1)
            input_data_transformed=input_data[i].view(-1, dim) if i==0 else torch.cat(out.outputs+[input_data[i].view(-1, dim)],dim=-1)
            q = current_vae.encoder.encode(input_data_transformed)
            mu = current_vae._reparam_layers['mu'](q)
            logvar = current_vae._reparam_layers['var'](q)
            zeta = current_vae.reparameterize(mu, logvar)
            zeta_transformed=zeta
            for output in out.outputs:
                zeta_transformed=torch.cat([zeta_transformed,output],dim=-1)
            output_data = current_vae.decoder.decode(zeta_transformed)
            out.outputs.append(output_data)
            out.mus.append(mu)
            out.logvars.append(logvar)
            
        return out

    def generate_samples(self):

        outputs=[]
        for i,it_dim in enumerate(self._flat_input_size):
            rnd_input=torch.randn((config.n_generate_samples,self._latent_dimensions))
            rnd_input_cat=torch.cat([rnd_input]+ outputs, dim=1)
            output = self._autoencoders[i].decoder.decode(rnd_input_cat)
            output.detach()

            outputs.append(output)
        return outputs
    
    def reparameterize(self, mu, logvar):
        """ 
        Sample epsilon from the normal distributions. Return mu+epsilon*sqrt(var),
        corresponding to random sample from Gaussian with mean mu and variance var.
        """
        
        eps = torch.randn_like(mu)
        return mu + eps*torch.einput_datap(0.5 * logvar)

    def loss(self, input_data, fwd_out):

        total_loss=0
        for i,dim in enumerate(self._flat_input_size):
            #fwd out contains lists with each member being the output for one layer.
            #hence need to construct single OutputContainer object for each VAE
            #layer.
            #TODO how to index the OutputContainer lists, so we can call
            #fwdout[i] here?

            tmp_out=OutputContainer(
                output_data=fwd_out.outputs[i],
                mu=fwd_out.mus[i],
                logvar=fwd_out.logvars[i]
            )
            in_data=input_data[i].view(-1, dim)
            total_loss+=self._autoencoders[i].loss(in_data, tmp_out)
        return total_loss