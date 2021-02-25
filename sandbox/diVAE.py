
"""
Discrete Variational Autoencoder Class Structures

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from utils.networks import HierarchicalEncoder,BasicEncoder,BasicDecoder
from models.rbm import RBM
from utils.distributions import Bernoulli

from copy import copy
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(1)

# Base Class for all AutoEncoder models
class AutoEncoderBase(nn.Module):
    def __init__(self, input_dimension=None, activation_fct=None, config=None, **kwargs):
        super(AutoEncoderBase,self).__init__(**kwargs)
        if isinstance(input_dimension,list):
            assert len(input_dimension)>0, "Input dimension not defined, needed for model structure"
        else:
            assert input_dimension>0, "Input dimension not defined, needed for model structure"
        assert config is not None, "Config not defined"
        assert config.model.n_latent_nodes is not None and config.model.n_latent_nodes>0, "Latent dimension must be >0"
        
        self._model_type=None
        self._config=config
        self._latent_dimensions=config.model.n_latent_nodes
        logger.warning("Taking all input dimensions for sVAE. Only first for other models.")
        self._input_dimension=input_dimension if self._config.model.model_type=="sVAE" else input_dimension[0]
        self._activation_fct=activation_fct
        self._dataset_mean=None

    def type(self):
        return self._model_type

    def _create_encoder(self):
        raise NotImplementedError

    def _create_decoder(self):
        raise NotImplementedError
    
    def __repr__(self):
        parameter_string="\n".join([str(par) for par in self.__dict__.items()])
        return parameter_string
    
    def forward(self, x):
        raise NotImplementedError

    def print_model_info(self):
        for par in self.__dict__.items():
            if isinstance(par,torch.Tensor):
                logger.info(par.shape)
            else:
                logger.info(par)

    # def set_dataset_mean(self,mean):
    #     self._dataset_mean=mean
    #TODO make this a getter setter thing
    def set_dataset_mean(self,mean):
        self._dataset_mean=mean[0] if isinstance(mean,list) else mean

# Autoencoder implementation
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
       
        dec_node_list=[self._latent_dimensions]+self._config.model.decoder_hidden_nodes+[self._input_dimension]

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

    def forward(self, x):
        zeta = self.encoder.encode(x.view(-1,self._input_dimension))
        output_data = self.decoder.decode(zeta)
        return output_data, zeta
    
    def loss(self, input_data, output_data):
        return self._loss_fct(output_data, input_data.view(-1,self._input_dimension), reduction='sum')

#Adds VAE specific reparameterisation, loss and forward call to AutoEncoder framework
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
        
        dec_node_list=[self._latent_dimensions]+self._config.model.decoder_hidden_nodes+[self._input_dimension]

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
    
    #TODO Is this the correct sampling procedure?
    # Draw a rnd var z~N[0,1] and feed it through the decoder alone? 
    def generate_samples(self,n_samples=100):
        """ 
        Similar to fwd. only skip encoding part...
        """
        rnd_input=torch.randn((n_samples,self._reparam_nodes[1]))
        zeta=rnd_input 
        # rnd_input=torch.where((rnd_input>0.5),torch.ones(rnd_input.size()),torch.zeros(rnd_input.size()))
        # print(rnd_input) 
        # output, mu, logvar, zeta=self.forward(rnd_input)
        # mu = self._reparam_layers['mu'](rnd_input)
        # logvar = self._reparam_layers['var'](rnd_input)
        # zeta = self.reparameterize(mu, logvar)
        output = self.decoder.decode(zeta)
        return output

    def loss(self, x, output_data, mu, logvar):
        logger.debug("VAE Loss")
        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(output_data, x.view(-1, self._input_dimension), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
                            
    def forward(self, x):
        z = self.encoder.encode(x.view(-1, self._input_dimension))
        mu = self._reparam_layers['mu'](z)
        logvar = self._reparam_layers['var'](z)
        zeta = self.reparameterize(mu, logvar)
        output_data = self.decoder.decode(zeta)
        return output_data, mu, logvar, zeta

#VAE with a hierarchical posterior modelled by encoder
#samples drawn from gaussian
class HierarchicalVAE(AutoEncoder):
    def __init__(self, **kwargs):
        super(HierarchicalVAE, self).__init__(**kwargs)
   
        self._model_type="HiVAE"

        self._reparamNodes=(self._config.model.n_encoder_layer_nodes,self._latent_dimensions)  

        self._decoder_nodes=[]
        dec_node_list=[(int(self._latent_dimensions*self._config.model.n_latent_hierarchy_lvls))]+self._config.model.decoder_hidden_nodes+[self._input_dimension]

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
            input_dimension=self._input_dimension,
            n_latent_hierarchy_lvls=self._config.model.n_latent_hierarchy_lvls,
            n_latent_nodes=self._latent_dimensions,
            n_encoder_layer_nodes=self._config.model.n_encoder_layer_nodes,
            n_encoder_layers=self._config.model.n_encoder_layers,
            skip_latent_layer=True)

    #TODO should this be part of encoder?
    def _create_reparameteriser(self,act_fct=None):
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
        
    def loss(self, x, output_data, mu_list, logvar_list):
        logger.debug("loss")
        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(output_data, x.view(-1, self._input_dimension), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        mu=torch.cat(mu_list,axis=1)
        logvar=torch.cat(logvar_list,axis=1)
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
                            
    def forward(self, x):
        data=x.view(-1, self._input_dimension)
        # x_enc_logits=[]
        zeta_list=[]
        mu_list=[]
        logvar_list=[]
        lvl=0
        for hierarchy in self.encoder._networks:
            indata=torch.cat([data,*zeta_list],axis=1)

            #apply activation fct as the hierarchicalposterior gives back
            #identity operation on last layer
            x_enc_hierarchy_logits=self.encoder.activation_fct(hierarchy(indata))
            # x_enc_logits.append(x_enc_hierarchy_logits)

            mu = self.reparameteriser['mu_'+str(lvl)](x_enc_hierarchy_logits)
            logvar = self.reparameteriser['var_'+str(lvl)](x_enc_hierarchy_logits)
            mu_list.append(mu)
            logvar_list.append(logvar)
            zeta = self.reparameterize(mu, logvar)
            zeta_list.append(zeta)
            lvl+=1    

        zeta_concat=torch.cat(zeta_list,axis=1)
        output_data = self.decoder.decode(zeta_concat)
        return output_data, mu_list, logvar_list, zeta_list

class DiVAE(AutoEncoderBase):
    def __init__(self, **kwargs):
        super(DiVAE, self).__init__(**kwargs)
        self._model_type="DiVAE"

        # self._decoder_nodes=[(self._latent_dimensions,128),]
        #TODO can this be done through inheritance?
        self._decoder_nodes=[]
        dec_node_list=[(int(self._latent_dimensions*self._config.model.n_latent_hierarchy_lvls))]+self._config.model.decoder_hidden_nodes+[self._input_dimension]

        for num_nodes in range(0,len(dec_node_list)-1):
            nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
            self._decoder_nodes.append(nodepair)

        #TODO change names globally
        #TODO one wd factor for both SimpleDecoder and encoder
        self.weight_decay_factor=self._config.engine.weight_decay_factor
        
        #ENCODER SPECIFICS
    
        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hierarchy level an output layer is formed.
        self.n_latent_hierarchy_lvls=self._config.model.n_latent_hierarchy_lvls

        #number of latent nodes in the prior - output nodes for each level of
        #the hierarchy. Also number of input nodes to the SimpleDecoder, first layer
        self.n_latent_nodes=self._config.model.n_latent_nodes

        #each hierarchy has NN with n_encoder_layers_enc layers
        #number of deterministic nodes in each encoding layer. These layers map
        #input to the latent layer. 
        self.n_encoder_layer_nodes=self._config.model.n_encoder_layer_nodes
        
        #TODO this could be solved more elegantly. FOr example replace
        #"skip_latent layer" with something actually useful 
        # assert self.n_latent_nodes==self.n_encoder_layer_nodes, "Number of nodes in last det encoder layer must be the same as num latent unit"

        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.n_encoder_layers=self._config.model.n_encoder_layers

        self._train_bias=None

    def set_train_bias(self):
        # self.train_bias = -np.log(1. / np.clip(self.config_train['mean_x'], 0.001, 0.999) - 1.).astype(np.float32)
        clipped_mean=torch.clamp(self._dataset_mean,0.001,0.999).detach()
        self._train_bias=-torch.log(1/clipped_mean-1)
        # self._train_bias.detach()
        return

    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.prior=self._create_prior()
        self.decoder=self._create_decoder()
        # print(self.encoder)
        # print(self.decoder)
        # print(self.prior)
        # exit()
        return
    
    def _create_encoder(self):
        logger.debug("ERROR _create_encoder dummy implementation")
        return HierarchicalEncoder(
            input_dimension=self._input_dimension,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            n_encoder_layer_nodes=self.n_encoder_layer_nodes,
            n_encoder_layers=self.n_encoder_layers,
            skip_latent_layer=False)

    def _create_decoder(self):
        logger.debug("_create_decoder")
        #Identity output_activation_fct, as this sigmoid is called manually in forward()
        return BasicDecoder(node_sequence=self._decoder_nodes, activation_fct=self._activation_fct, output_activation_fct=nn.Identity())

    def _create_prior(self):
        logger.debug("_create_prior")
        num_rbm_nodes_per_layer=self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions//2
        return RBM(n_visible=num_rbm_nodes_per_layer,n_hidden=num_rbm_nodes_per_layer)
   
    def weight_decay_loss(self):
        #TODO
        logger.debug("ERROR weight_decay_loss NOT IMPLEMENTED")
        return 0
    
    def train_rbm(self):
        self.prior.train_sampler()
        return

    def loss(self, in_data, output, output_activations, output_distribution, posterior_distribution,posterior_samples):
        logger.debug("loss")

        #1) total_kl = .prior.kl_dist_from(posterior, post_samples,
        #   is_training)
        #KLD
        # checked on 201023 - various TODOs and question marks....        
        kl_loss=self.kl_divergence(posterior_distribution,posterior_samples)
        # exit()
        #2)         # expected log prob p(x| z)
        # cost = - output_dist.log_prob_per_var(input)
        # cost = tf.reduce_sum(cost, axis=1)
        
        #TODO this alright? sign, softplus, DWAVE vs torch source implementation
        #this returns a matrix 100x784 (samples times var)
        #           output distribution is Bernoulli at the end of VAE. Input
        #           distribution is data.
        ae_loss_matrix=-output_distribution.log_prob_per_var(in_data.view(-1, self._input_dimension))
        #loss is the sum of all variables (pixels) per sample (event in batch)
        ae_loss=torch.sum(ae_loss_matrix,1)

        #3) weight decay loss
        # enc_wd_loss = self.encoder.get_weight_decay()
        # dec_wd_loss = self.decoder.get_weight_decay()
        # prior_wd_loss = self.prior.get_weight_decay() if isinstance(self.prior, RBM) else 0
        # wd_loss = enc_wd_loss + dec_wd_loss + prior_wd_loss

        #4) neg elbo warm-up idea kl-term
        # kl_coeff = self.kl_coeff_annealing(is_training)
        # neg_elbo_per_sample = kl_coeff * total_kl + cost
        # neg_elbo = tf.reduce_mean(neg_elbo_per_sample, name='neg_elbo')

        # print(ae_loss)
        neg_elbo_per_sample=ae_loss+kl_loss

        #the mean of the elbo over all samples is taken as measure for loss
        neg_elbo=torch.mean(neg_elbo_per_sample)    

        #include the weight decay regularisation in the loss to penalise complexity
        loss=neg_elbo#+weight_decay_loss  
        return loss


    def kl_div_prior_gradient(self, posterior_logits, posterior_binary_samples):
        #DVAE Eq11 - gradient of prior
        #gradient of the KLD between posterior and prior wrt to prior
        #parameters theta, i.e. generative model parameters.
        """
        Integrated gradient of the KL-divergence between a hierarchical approximating posterior and an RBM prior.
        When differentiated, this gives the gradient with respect to the RBM prior.
        The last layer in the hierarchy of the approximating posterior can be Rao-Blackwellized.
        All previous layers must be sampled, or the J term is incorrect; the h term can accommodate probabilities
        For the rbm_samples, one side can be Rao-Blackwellized
        Args:
            posterior_logits:         list of approx. post. logits.
            posterior_binary_samples: list of approx. post. samples with last layer marginalized.
            rbm_samples:                rbm samples

        Returns:
            kld_per_sample:             the KL tensor containing proper gradients for prior
        """
        #logits to probabilities
        posterior_probs=torch.sigmoid(posterior_logits)
        positive_probs=posterior_probs.detach()
        
        #samples from posterior are labelled positive
        positive_samples=posterior_binary_samples.detach()

        n_split=positive_samples.size()[1]//2
        positive_samples_left,positive_samples_right=torch.split(positive_samples,split_size_or_sections=int(n_split),dim=1)
     
       #-z_left^t J z_right
        pos_first_term=torch.matmul(positive_samples_left,self.prior.get_weights())*positive_samples_right
       
        rbm_bias_left=self.prior.get_visible_bias()
        rbm_bias_right=self.prior.get_hidden_bias()
        rbm_bias=torch.cat([rbm_bias_left,rbm_bias_right])#self._h
        
        #this gives [42,400] size
        #- z^t h
        #TODO this uses positive probs. Should it not use positive samples?
        # FIXME an indication are the negative ones where samples are used! On
        #other hand this is the only place this this used
        pos_sec_term=positive_probs*rbm_bias
        # pos_sec_term=positive_samples*rbm_bias

       # Energy = -z_left^t J z_right - z^t h
        pos_kld_per_sample=-(torch.sum(pos_first_term,axis=1)+torch.sum(pos_sec_term,axis=1))
        #samples from rbm are labelled negative

        #rbm_samples Tensor("zeros:0", shape=(200, 200), dtype=float32)
        #this returns the full RBM set: left and right nodes concatenated

        #TODO What are these samples here?
        #TODO what's the impact of doing gibbs sampling here? does this make
        #sense?
        rbm_samples=self.prior.get_samples_kld(approx_post_samples=positive_samples_left,n_gibbs_sampling_steps=1)
        negative_samples=rbm_samples.detach()

        # print(self.prior.get_weights())
        n_split=negative_samples.size()[1]//2
        negative_samples_left,negative_samples_right=torch.split(negative_samples,split_size_or_sections=int(n_split),dim=1)
        neg_first_term=torch.matmul(negative_samples_left,self.prior.get_weights())*negative_samples_right
        
        #FIXME see above, the positive case looks different. Why?
        neg_sec_term=negative_samples*rbm_bias
        neg_kld_per_sample=(torch.sum(neg_first_term,axis=1)+torch.sum(neg_sec_term,axis=1))
        
        kld_per_sample=pos_kld_per_sample+neg_kld_per_sample

        return kld_per_sample

    def kl_div_posterior_gradient(self, posterior_logits, posterior_binary_samples):
        #DVAE Eq12
        #gradient of the KLD between posterior and prior wrt to posterior
        #parameters phi
        """
        Integrated gradient of the KL-divergence between a hierarchical approximating posterior and an RBM prior.
        When differentiated, this gives the gradient with respect to the approximating posterior
        Approximating posterior is q(z = 1) = sigmoid(logistic_input).  Equivalently, E_q(z) = -logistic_input * z, with
        p(z) = e^-E_q / Z_q
        Args:
            posterior_logits: posterior_logits        list of approx. post. logits.
            approx_post_binary_samples: posterior_binary_samples list of approx. post. samples with last layer marginalized.

        Returns:
            kld_per_sample:             the KL tensor containing proper gradients for aapprox. post.
        """
        
        logger.debug("kl_div_posterior_gradient")
        posterior_upper_bound = 0.999*torch.ones_like(posterior_logits)
        #logits to probabilities
        posterior_probs=torch.min(posterior_upper_bound, torch.sigmoid(posterior_logits))
         
        n_split=int(posterior_binary_samples.size()[1]//2)
        #binarised samples from posterior to RBM layers
        rbm_samples_left,rbm_samples_right=torch.split(posterior_binary_samples,split_size_or_sections=n_split,dim=1)

        #the following prepares the variables in the calculation in tehir format
        rbm_bias_left=self.prior.get_visible_bias()
        rbm_bias_right=self.prior.get_hidden_bias()

        rbm_bias=torch.cat([rbm_bias_left,rbm_bias_right])#self._h
        rbm_weight=self.prior.get_weights()#self._J

        # this is transposed, so we multiply what we call "right hand" ("hidden layer")
        # samples with right rbm nodes
        # rbm_weight_t=torch.transpose(rbm_weight,0,1)#self._J
        
        rbm_activation_right=torch.matmul(rbm_samples_right,rbm_weight.t())
        rbm_activation_left=torch.matmul(rbm_samples_left,rbm_weight)

        #corresponds to samples_times_J
        rbm_activation=torch.cat([rbm_activation_right,rbm_activation_left],1)
        
        #TODO what is this scaling factor?
        #[400,400] 
        hierarchy_scaling= (1.0 - posterior_binary_samples) / (1.0 - posterior_probs)
        hierarchy_scaling_left,hierarchy_scaling_right=torch.split(hierarchy_scaling, split_size_or_sections=int(n_split),dim=1)
        
        #TODO why does this happen? This seems to scale only the left side of
        #the RBM. Th right side is replaced with ones.
        hierarchy_scaling_with_ones=torch.cat([hierarchy_scaling_left,torch.ones(hierarchy_scaling_right.size())],axis=1)
        
        with torch.no_grad():
            undifferentiated_component=posterior_logits-rbm_bias-rbm_activation*hierarchy_scaling_with_ones
            undifferentiated_component=undifferentiated_component.detach()
        
        kld_per_sample = torch.sum(undifferentiated_component * posterior_probs, dim=1)

        return kld_per_sample

    def kl_divergence(self, posterior_distribution , posterior_samples):
        logger.debug("kl_divergence")
        #posterior_distribution: distribution with logits from each hierarchy level/layer
        #posterior_samples: reparameterised output of posterior_distribution
        if len(posterior_distribution)>1 and self.training:

            logit_list=[]
            samples_marginalised=[]
            for lvl in range(len(posterior_distribution)):

                current_post_dist=posterior_distribution[lvl]
                current_post_samples=posterior_samples[lvl]

                logits=torch.clamp(current_post_dist.logits,min=-88,max=88)
                logit_list.append(logits)
            
                if lvl==len(posterior_distribution)-1:
                    samples_marginalised.append(torch.sigmoid(logits))
                else:
                    zero_mask=torch.zeros(current_post_samples.size())
                    one_mask=torch.ones(current_post_samples.size())
                    # posterior_distribution_sample>0.5,
                    #TODO  what's happening here? some kind of binarisation for
                    #the RBM? Is 0 correct or should my data be standardised
                    #before that?
                    #FIXME I have doubts that this is correct...
                    #DVAE Eq11 - gradient of AE model. Does this explain it?
                    post_sample_marginalised=torch.where(current_post_samples>0.0,one_mask,zero_mask)
                    samples_marginalised.append(post_sample_marginalised)

            logits_concat=torch.cat(logit_list,1)
            samples_marginalised_concat=torch.cat(samples_marginalised,1)

            kl_div_posterior_distribution=self.kl_div_posterior_gradient(
                posterior_logits=logits_concat,
                posterior_binary_samples=samples_marginalised_concat)
                
            kl_div_prior=self.kl_div_prior_gradient(
                posterior_logits=logits_concat,
                posterior_binary_samples=samples_marginalised_concat)  #DVAE Eq11 - gradient of prior   
            kld=kl_div_prior+kl_div_posterior_distribution 
            return kld
        else: # either this posterior only has one latent layer or we are not looking at training
            # #this posterior is not hierarchical - a closed analytical form for the KLD term can be constructed
            # #the mean-field solution (n_latent_hierarchy_lvls == 1) reduces to log_ratio = 0.
            # logger.debug("kld for evaluation/training of one layer posterior")
            return 0

    def generate_samples_per_gibbs(self, init_left_samples=None, init_right_samples=None, steps=20):
        output_per_step=[]
        n_samples=5
        for step in range(0,steps):
            prior_samples=[]
            for i in range(0,n_samples):
                prior_sample = self.prior.get_samples_per_gibbs(
                    init_left_samples=init_left_samples.detach(),
                    init_right_samples=init_right_samples.detach(),
                    n_gibbs_sampling_steps=100
                )
                prior_sample = torch.cat(prior_sample)
                prior_samples.append(prior_sample)

            prior_samples=torch.stack(prior_samples)
            # prior_samples = tf.slice(prior_samples, [0, 0], [num_samples, -1])
            output_activations = self.decoder.decode(prior_samples)
            output_activations = output_activations+self._train_bias
            output_distribution = Bernoulli(logit=output_activations)
            output=torch.sigmoid(output_distribution.logits)
            output_per_step.append(output)
        return output_per_step


    def generate_samples(self, n_samples=100):
        """ It will randomly sample from the model using ancestral sampling. It first generates samples from p(z_0).
        Then, it generates samples from the hierarchical distributions p(z_j|z_{i < j}). Finally, it forms p(x | z_i).  
        
         Args:
             num_samples: an integer value representing the number of samples that will be generated by the model.
        """
        		
		#how many times should ancestral sampling be run
		#n_samples
        prior_samples=[]
        for i in range(0,n_samples):
            prior_sample = self.prior.get_samples(
                n_latent_nodes=self.n_latent_nodes,
                n_gibbs_sampling_steps=100, 
                sampling_mode="gibbs_ancestral")
            prior_sample = torch.cat(prior_sample)
            prior_samples.append(prior_sample)
        prior_samples=torch.stack(prior_samples)
        # prior_samples = tf.slice(prior_samples, [0, 0], [num_samples, -1])
        output_activations = self.decoder.decode(prior_samples)
        output_activations = output_activations+self._train_bias
        output_distribution = Bernoulli(logit=output_activations)
        output=torch.sigmoid(output_distribution.logits)
        # output_activations[0] = output_activations[0] + self.train_bias
        # output_dist = FactorialBernoulliUtil(output_activations)
        # output_samples = tf.nn.sigmoid(output_dist.logit_mu)
        # print("--- ","end VAE::generate_samples()")
        return output            

    def forward(self, in_data):
        logger.debug("forward")
        #TODO data prep - study if this does good things
        in_data_centered=in_data.view(-1, self._input_dimension)-self._dataset_mean
        #Step 1: Feed data through encoder 
        posterior_distributions, posterior_samples = self.encoder.hierarchical_posterior(in_data_centered)
        posterior_samples_concat=torch.cat(posterior_samples,1)
        #Step 2: take samples zeta and reconstruct output with decoder
        output_activations = self.decoder.decode(posterior_samples_concat)
        #TODO data prep
        output_activations = output_activations+self._train_bias
        output_distribution = Bernoulli(logit=output_activations)
        output = torch.sigmoid(output_distribution.logits)
        return output, output_activations, output_distribution, \
            posterior_distributions, posterior_samples


if __name__=="__main__":
    logger.debug("Testing Model Setup") 
    # model=VAE()
    # model=DiVAE()
    model=HierarchicalVAE()
    print(model.encoder)
    logger.debug("Success")
    pass