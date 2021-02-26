
"""
Variational Autoencoder Model with hierarchical encoder

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""
import torch
from torch import nn
from models.autoencoder import AutoEncoder

from utils.networks import HierarchicalEncoder

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

#VAE with a hierarchical posterior modelled by encoder
#samples drawn from gaussian
class HierarchicalVAE(AutoEncoder):
    def __init__(self, **kwargs):
        super(HierarchicalVAE, self).__init__(**kwargs)
   
        self._model_type="HiVAE"

        self._reparamNodes=(self._config.model.n_encoder_layer_nodes,self._latent_dimensions)  

        self._decoder_nodes=[]

        dec_hidden_node_list=list(self._config.model.decoder_hidden_nodes)
        dec_node_list=[(int(self._latent_dimensions*self._config.model.n_latent_hierarchy_lvls))]+dec_hidden_node_list+[self._flat_input_size]

        for num_nodes in range(0,len(dec_node_list)-1):
            nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
            self._decoder_nodes.append(nodepair)

    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.reparameteriser=self._create_reparameteriser()
        self.decoder=self._create_decoder()
        return

    def _create_encoder(self,act_fct=None):
        logger.debug("_create_encoder")
        return HierarchicalEncoder(
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self._config.model.n_latent_hierarchy_lvls,
            n_latent_nodes=self._latent_dimensions,
            n_encoder_layer_nodes=self._config.model.n_encoder_layer_nodes,
            n_encoder_layers=self._config.model.n_encoder_layers,
            skip_latent_layer=True)
    
    
    def _create_reparameteriser(self):
        """Create layers fopr reparameterisation specific to this model. I.e.
hierarchical means mu and variances var for the gaussians learned in each
hierarchy layer.

        Returns:
            [type]: [description]
        """
        logger.debug("ERROR _create_encoder dummy implementation")
        hierarchical_repara_layers=nn.ModuleDict()
        for lvl in range(self._config.model.n_latent_hierarchy_lvls):
            hierarchical_repara_layers['mu_'+str(lvl)]=nn.Linear(self._reparamNodes[0],self._reparamNodes[1])
            hierarchical_repara_layers['var_'+str(lvl)]=nn.Linear(self._reparamNodes[0],self._reparamNodes[1])
        return hierarchical_repara_layers

    def reparameterize(self, mu, logvar):
        """ Sample from the normal distributions corres and return var * samples + mu
        """
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
        
    def loss(self, input_data, fwd_out):
        # output_data, mu_list, logvar_list=fwd_out.
        logger.debug("loss")
        # Autoencoding term
        auto_loss = nn.functional.binary_cross_entropy(fwd_out.output_data, input_data.view(-1, self._flat_input_size), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        mu=torch.cat(fwd_out.mu_list,axis=1)
        logvar=torch.cat(fwd_out.logvar_list,axis=1)
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
                            
    def forward(self, input_data):
        #see definition for explanation
        out=self._output_container.clear()

        out.zeta_list=[]
        out.mu_list=[]
        out.logvar_list=[]

        data=input_data.view(-1, self._flat_input_size)
        lvl=0
        for hierarchy in self.encoder._networks:
            indata=torch.cat([data,*out.zeta_list],axis=1)

            #apply activation fct as the hierarchical posterior gives back
            #identity operation on last layer
            q_enc_hierarchy_logits=self.encoder.activation_fct(hierarchy(indata))

            mu = self.reparameteriser['mu_'+str(lvl)](q_enc_hierarchy_logits)
            logvar = self.reparameteriser['var_'+str(lvl)](q_enc_hierarchy_logits)
            out.mu_list.append(mu)
            out.logvar_list.append(logvar)
            zeta = self.reparameterize(mu, logvar)
            out.zeta_list.append(zeta)
            lvl+=1    

        zeta_concat=torch.cat(out.zeta_list,axis=1)
        out.output_data = self.decoder.decode(zeta_concat)

        return out

if __name__=="__main__":
    logger.info("Running hierarchicalVAE.py directly") 
    model=HierarchicalVAE()
    print(model)
    logger.info("Success")