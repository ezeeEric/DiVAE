"""
VAE for calo data adapted from ATL-SOFT-PUB-2018-001
"""

# Torch imports
import torch
from torch.nn import ReLU, ELU, MSELoss
from torch.nn.modules.activation import Sigmoid

# DiVAE.models imports
from models.autoencoders.variationalAE import VariationalAutoEncoder

from DiVAE import logging
logger = logging.getLogger(__name__)

class ATLASVAE(VariationalAutoEncoder):
    
    def __init__(self, **kwargs):
        super(ATLASVAE, self).__init__(**kwargs)
        self._model_type = "ATLASVAE"
        self._output_activation_fct = Sigmoid()
        #self._output_activation_fct = ReLU()
        self._output_loss = MSELoss(reduction="sum")
        
        #ELU chosen in paper
        self._activation_fct=ELU()

        #our VAE is conditioned on the energy, need to add a node for this
        #feature, i.e. the first encoder layer and first decoder layer
        self._encoder_nodes[0]=(self._encoder_nodes[0][0]+1,self._encoder_nodes[0][1])
        self._decoder_nodes[0]=(self._decoder_nodes[0][0]+1,self._decoder_nodes[0][1])

        #these parameters should be optimised in a hyperparameter scan
        self.w_reco=self._config.model.w_reco
        self.w_kl=self._config.model.w_kl
        self.w_etot=self._config.model.w_etot
        self.w_ei=list(self._config.model.w_ei)
        
    def forward(self, x, energy):
        logger.debug("forward")
        
        out=self._output_container.clear()       

        #normalise to deposited E
        x_norm=torch.div(x,energy)
        
        #append energy to normalised tensor
        x_norm=torch.cat([x_norm,energy],dim=1).float()

        z = self.encoder(x_norm)

        out.mu = self._reparam_layers['mu'](z)
        out.logvar = self._reparam_layers['var'](z)

        zeta_tmp=self.reparameterize(out.mu, out.logvar)
        out.zeta = torch.cat([zeta_tmp,energy],dim=1).float()

        out.output_activations = self._output_activation_fct(self.decoder(out.zeta))
        out.output_activations = torch.mul(out.output_activations, energy).float()
        return out
    
    def loss(self, input_data, fwd_out, in_dim=None):
        logger.debug("VAE Loss")
        # Autoencoding term
        loss_reco=self._output_loss(fwd_out.output_activations, input_data)
        
        #KL loss term, assuming Gaussian prior
        loss_kl=0.5 * torch.sum(1 + fwd_out.logvar - fwd_out.mu.pow(2) - torch.exp(fwd_out.logvar)).double()
        
        #total energy loss
        #avoid numerical instability by setting lower bound (used as divisor further below)
        e_tot=max(torch.sum(input_data),0.001)
        e_tot_reco=max(torch.sum(fwd_out.output_activations),0.001)

        loss_etot=torch.abs(e_tot-e_tot_reco) 
        loss_etot=torch.sum(loss_etot)

        #loss energy per layer
        e_layer=[]
        e_layer_reco=[]

        #this takes the flat, unrolled input and splits it into the layer chunks
        for it_image_layer in torch.split(input_data, in_dim, dim=1):
            e_layer.append(torch.sum(it_image_layer))
            
        for it_out_layer in torch.split(fwd_out.output_activations, in_dim, dim=1):
            e_layer_reco.append(torch.sum(it_out_layer))

        loss_ei=0
        for l in range(0,len(e_layer)):
            loss_ei+=self.w_ei[l]*torch.abs(e_layer[l]/e_tot-e_layer_reco[l]/e_tot_reco)

        loss=self.w_reco*loss_reco-self.w_kl*loss_kl+self.w_etot*loss_etot+loss_ei
        
        return {"loss":loss, "reco_loss": loss_reco, "loss_kl": loss_kl, "loss_etot": loss_etot, "loss_ei":loss_ei}
    
    def generate_samples(self, target_energies=None):
        """
        generate_samples()
        """ 
        #uniform distributed energies between [0,100) for the generated jets
        if not target_energies:
            target_energies=100*torch.rand((self._config.engine.n_valid_batch_size,1))
        
        samples = []
        for s in range(0,self._config.engine.n_valid_batch_size):
            energy=target_energies[s].unsqueeze(-1).float()
            rnd_input=torch.randn((1,self._reparam_nodes[1]),dtype=torch.float32)
            rnd_input_cat=torch.cat([rnd_input,energy], dim=1).float()
            output = self._output_activation_fct(self.decoder(rnd_input_cat))
            output = torch.mul(output, energy)
            samples.append(output)

        samples_cat=torch.cat(samples, dim=0)
        return samples_cat