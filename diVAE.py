# -*- coding: utf-8 -*-
"""
Discrete Variational Autoencoder Class Structures

Author: Eric Drechsler (eric_drechsler@sfu.ca)

Based on work from Olivia di Matteo.
"""
#TODO can go after debug
import math

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from networks import Decoder,HierarchicalEncoder,SimpleEncoder,SimpleDecoder,BasicEncoder,BasicDecoder
from rbm import RBM
from distributions import Bernoulli

from copy import copy
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(1)

# Base Class for all AutoEncoder models
class AutoEncoderBase(nn.Module):
    def __init__(self, input_dimension=None, activation_fct=None, config=None, **kwargs):
        super(AutoEncoderBase,self).__init__(**kwargs)
        
        assert input_dimension is not None and input_dimension>0, "Input dimension not defined, needed for model structure"
        assert config is not None, "Config not defined"
        assert config.num_latent_units is not None and config.num_latent_units>0, "Latent dimension must be >0"
        
        self._type=None
        self._config=config
        self._latent_dimensions=config.num_latent_units
        self._input_dimension=input_dimension
        self._activation_fct=activation_fct

    def type(self):
        return self._type

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
            logger.info(par)

# Autoencoder implementation
class AutoEncoder(AutoEncoderBase):

    def __init__(self, **kwargs):
        super(AutoEncoder,self).__init__(**kwargs)
        self._type="AE"

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

        #only works if x_true, x_recon in [0,1]
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
        x_recon = self.decoder.decode(zeta)
        return x_recon, zeta
    
    def loss(self, x_true, x_recon):
        return self._loss_fct(x_recon, x_true.view(-1,self._input_dimension), reduction='sum')

#Adds VAE specific reparameterisation, loss and forward call to AutoEncoder framework
class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        
        self._type="VAE"

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

#VAE with a hierarchical posterior modelled by encoder
#samples still drawn from gaussian
class HiVAE(AutoEncoderBase):
    def __init__(self, **kwargs):
        super(HiVAE, self).__init__(**kwargs)
        
        self._type="HiVAE"

        self._reparamNodes=(200,self._latent_dimensions)   
        self._decoder_nodes=[(int(self._latent_dimensions*self._config.num_latent_hierarchy_levels),200),(200,784)]

        self.reparameteriser=self._create_reparameteriser()

        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
    
    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.reparameteriser=self._create_reparameteriser()
        self.decoder=self._create_decoder()
        return

    def _create_encoder(self,act_fct=None):
        encoder=HierarchicalEncoder(
            num_latent_units=self._config.num_latent_units,
            num_latent_hierarchy_levels=self._config.num_latent_hierarchy_levels,
            use_gaussian=True)
        return encoder

    def _create_reparameteriser(self,act_fct=None):
        logger.debug("ERROR _create_encoder dummy implementation")
        hierarchical_repara_layers=nn.ModuleDict()
        for lvl in range(self._config.num_latent_hierarchy_levels):
            hierarchical_repara_layers['mu_'+str(lvl)]=nn.Linear(self._reparamNodes[0],self._reparamNodes[1])
            hierarchical_repara_layers['var_'+str(lvl)]=nn.Linear(self._reparamNodes[0],self._reparamNodes[1])
        return hierarchical_repara_layers

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return SimpleDecoder(node_sequence=self._decoder_nodes, activation_fct=self._activation_fct, output_activation_fct=nn.Sigmoid())
        
    def reparameterize(self, mu, logvar):
        """ Sample from the normal distributions corres and return var * samples + mu
        """
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
        
    def loss(self, x, x_recon, mu_list, logvar_list):
        logger.debug("loss")
        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        mu=torch.cat(mu_list,axis=1)
        logvar=torch.cat(logvar_list,axis=1)
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
                            
    def forward(self, x):
        data=x.view(-1, 784)
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
        x_recon = self.decoder.decode(zeta_concat)
        return x_recon, mu_list, logvar_list, zeta_list

class DiVAE(AutoEncoderBase):
    def __init__(self, n_hidden_units=256, **kwargs):
        super(DiVAE, self).__init__(**kwargs)
        self._type="DiVAE"

        self._reparamNodes=(128,self._latent_dimensions)  
        self._decoder_nodes=[(self._latent_dimensions,128),]
        self._outputNodes=(128,784)     

        self._n_hidden_units=n_hidden_units

        #TODO change names globally
        #configs from DWave
        #TODO one wd factor for both SimpleDecoder and encoder
        self.weight_decay_factor=1e-4
        
        #ENCODER SPECIFICS
        
        #number of hierarchy levels in encoder. This is the number of latent
        #layers. At each hiearchy level an output layer is formed.
        self.num_latent_hierarchy_levels=4

        #number of latent units in the prior - output units for each level of
        #the hierarchy. Also number of input nodes to the SimpleDecoder, first layer
        self.num_latent_units=100

        self.activation_fct=activation_fct
        #each hierarchy has NN with num_det_layers_enc layers
        #number of deterministic units in each encoding layer. These layers map
        #input to the latent layer. 
        self.num_det_units=200
        
        # number of deterministic layers in each conditional p(z_i | z_{k<i})
        self.num_det_layers=2 

        # for all layers except latent (output)
        self.activation_fct=nn.Tanh()

        self.encoder=self._create_encoder(act_fct=self.activation_fct)
        self.decoder=self._create_decoder()
        self.prior=self._create_prior()

    
    def _create_encoder(self,act_fct=None):
        logger.debug("ERROR _create_encoder dummy implementation")
        encoder=HierarchicalEncoder()
        return encoder

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return Decoder()

    def _create_prior(self):
        logger.debug("_create_prior")
        num_rbm_units_per_layer=self.num_latent_hierarchy_levels*self.num_latent_units//2
        return RBM(n_visible=num_rbm_units_per_layer,n_hidden=num_rbm_units_per_layer)
   
    def weight_decay_loss(self):
        #TODO
        logger.debug("ERROR weight_decay_loss NOT IMPLEMENTED")
        return 0

    def loss(self, in_data, output, output_activations, output_distribution, posterior_distribution,posterior_samples):
        logger.debug("loss")
        #TODO this alright? sign, softplus, DWAVE vs torch source implementation
        #this returns a matrix 100x784 (samples times var)
        ae_loss_matrix=-output_distribution.log_prob_per_var(in_data.view(-1, 784))
        #loss is the sum of all variables (pixels) per sample (event in batch)
        ae_loss=torch.sum(ae_loss_matrix,1)
        #KLD        
        kl_loss=self.kl_divergence(posterior_distribution,posterior_samples)
        # print(ae_loss)
        neg_elbo_per_sample=ae_loss+kl_loss
        # import sys
        # sys.exit()
        # if self.training:
        #     #kld per sample
        #     # total_kl = self.prior.kl_dist_from(posterior, post_samples, is_training)
        #     kl_loss=self.kl_divergence(hierarchical_posterior,posterior_samples)
        #     # # weight decay loss
        #     # enc_wd_loss = self.encoder.get_weight_decay()
        #     # dec_wd_loss = self.decoder.get_weight_decay()
        #     # prior_wd_loss = self.prior.get_weight_decay() if isinstance(self.prior, RBM) else 0
        #     weight_decay_loss=self.weight_decay_loss()
        #     neg_elbo_per_sample =  ae_loss+kl_loss 
        # else:
        #     #kld per sample
        #     # total_kl = self.prior.kl_dist_from(posterior, post_samples, is_training)
        #     #TODO during evaluation - why is KLD necessary?
        #     kl_loss=self.kl_divergence(hierarchical_posterior,posterior_samples)
        #     # # weight decay loss
        #     # enc_wd_loss = self.encoder.get_weight_decay()
        #     # dec_wd_loss = self.decoder.get_weight_decay()
        #     # prior_wd_loss = self.prior.get_weight_decay() if isinstance(self.prior, RBM) else 0
        #     neg_elbo_per_sample =  ae_loss+kl_loss 
        #     #since we are not training
        #     weight_decay_loss=0 
        #the mean of the elbo over all samples is taken as measure for loss
        neg_elbo=torch.mean(neg_elbo_per_sample)    
        #include the weight decay regularisation in the loss to penalise complexity
        loss=neg_elbo#+weight_decay_loss
        # return loss
        if math.isnan(loss):
            raise ValueError("Loss is NAN - KL Divergence diverged")       
        return loss

    def kl_div_prior_gradient(self, approx_post_logits, approx_post_binary_samples):
        #DVAE Eq12 - gradient of prior
        #that is, the gradient of the KL between approx posterior and priot wrt
        #phi, parameters of the RBM
        """
        Integrated gradient of the KL-divergence between a hierarchical approximating posterior and an RBM prior.
        When differentiated, this gives the gradient with respect to the RBM prior.
        The last layer in the hierarchy of the approximating posterior can be Rao-Blackwellized.
        All previous layers must be sampled, or the J term is incorrect; the h term can accommodate probabilities
        For the rbm_samples, one side can be Rao-Blackwellized
        Args:
            approx_post_logits:         list of approx. post. logits.
            approx_post_binary_samples: list of approx. post. samples with last layer marginalized.
            rbm_samples:                rbm samples

        Returns:
            kld_per_sample:             the KL tensor containing proper gradients for prior
        """
        #logits to probabilities
        approx_post_probs=torch.sigmoid(approx_post_logits)
        positive_probs=approx_post_probs.detach()
        
        #samples from posterior are labelled positive
        positive_samples=torch.cat(approx_post_binary_samples,1).detach()
        n_split=positive_samples.size()[1]//2
        positive_samples_left,positive_samples_right=torch.split(positive_probs,split_size_or_sections=int(n_split),dim=1)
        
        pos_first_term=torch.matmul(positive_samples_left,self.prior.get_weights())*positive_samples_right
       
        rbm_bias_left=self.prior.get_visible_bias()
        rbm_bias_right=self.prior.get_hidden_bias()
        rbm_bias=torch.cat([rbm_bias_left,rbm_bias_right])#self._h
        
        #this gives [42,400] size
        pos_sec_term=positive_probs*rbm_bias
        pos_kld_per_sample=-(torch.sum(pos_first_term,axis=1)+torch.sum(pos_sec_term,axis=1))
        #samples from rbm are labelled negative

        #rbm_samples Tensor("zeros:0", shape=(200, 200), dtype=float32)
        rbm_samples=self.prior.get_samples_kld(approx_post_samples=positive_samples_left)
        negative_samples=rbm_samples.detach()
        n_split=negative_samples.size()[1]//2
        negative_samples_left,negative_samples_right=torch.split(negative_samples,split_size_or_sections=int(n_split),dim=1)
        neg_first_term=torch.matmul(negative_samples_left,self.prior.get_weights())*negative_samples_right
        neg_sec_term=negative_samples*rbm_bias
        neg_kld_per_sample=(torch.sum(neg_first_term,axis=1)+torch.sum(neg_sec_term,axis=1))
        
        kld_per_sample=pos_kld_per_sample+neg_kld_per_sample
        return kld_per_sample

    def kl_div_posterior_gradient(self, approx_post_logits, approx_post_binary_samples):
        #DVAE Eq11 - gradient of AE model
        #that is, the gradient of the KL between approx posterior and priot wrt
        #theta, parameters of the AE model
        """
        Integrated gradient of the KL-divergence between a hierarchical approximating posterior and an RBM prior.
        When differentiated, this gives the gradient with respect to the approximating posterior
        Approximating posterior is q(z = 1) = sigmoid(logistic_input).  Equivalently, E_q(z) = -logistic_input * z, with
        p(z) = e^-E_q / Z_q
        Args:
            approx_post_logits:         list of approx. post. logits.
            approx_post_binary_samples: list of approx. post. samples with last layer marginalized.

        Returns:
            kld_per_sample:             the KL tensor containing proper gradients for aapprox. post.
        """
        
        logger.debug("kl_div_posterior_gradient")
        #TODO add cutoff
        # approx_post_upper_bound = 0.999
        # approx_post_probs = tf.minimum(approx_post_upper_bound, tf.sigmoid(approx_post_logits))

        #logits to probabilities
        approx_post_probs=torch.sigmoid(approx_post_logits)

        #samples from posterior to RBM layers
        #TODO better automatisation for this split?
        #TODO YES! see below torch.split
        samples_rbm_units_left=[]
        samples_rbm_units_right=[]
        for i in range(1,len(approx_post_binary_samples)+1):
            if i<=len(approx_post_binary_samples)/2:
                samples_rbm_units_left.append(approx_post_binary_samples[i-1])
            else:
                samples_rbm_units_right.append(approx_post_binary_samples[i-1])

        samples_rbm_units_left=torch.cat(samples_rbm_units_left,1)
        samples_rbm_units_right=torch.cat(samples_rbm_units_right,1)

        # print(samples_rbm_units_left.size())
        # torch.Size([42, 200])
        #the following prepares the variables in the calculation in tehir format
        rbm_bias_left=self.prior.get_visible_bias()
        rbm_bias_right=self.prior.get_hidden_bias()

        rbm_bias=torch.cat([rbm_bias_left,rbm_bias_right])#self._h

        #torch.Size([200, 201]) n_vis=200(left),n_hid=201 (right)
        rbm_weight=self.prior.get_weights()#self._J
        #torch.Size([201, 200]) n_vis=200(left),n_hid=201 (right)
        #this is transposed, so we multiply what we call "right hand" ("hidden layer")
        #samples with right rbm units
        rbm_weight_t=torch.transpose(rbm_weight,0,1)#self._J
        rbm_activation_right=torch.matmul(samples_rbm_units_right,rbm_weight_t)
        rbm_activation_left=torch.matmul(samples_rbm_units_left,rbm_weight)
        #corresponds to samples_times_J
        rbm_activation=torch.cat([rbm_activation_right,rbm_activation_left],1)

        hierarchy_scaling= (1.0 - torch.cat(approx_post_binary_samples,1)) / (1.0 - approx_post_probs)
        n_split=hierarchy_scaling.size()[1]//2
        hierarchy_scaling_left,hierarchy_scaling_right=torch.split(hierarchy_scaling,split_size_or_sections=int(n_split),dim=1)
        #TODO why does this happen?
        hierarchy_scaling_with_ones=torch.cat([hierarchy_scaling_left,torch.ones(hierarchy_scaling_right.size())],axis=1)

        undifferentiated_component=approx_post_logits-rbm_bias-rbm_activation*hierarchy_scaling_with_ones
        undifferentiated_component=undifferentiated_component.detach()
        
        kld_per_sample = torch.sum(undifferentiated_component * approx_post_probs, dim=1)
        
        return kld_per_sample

    def kl_divergence(self, posterior , posterior_samples):
        logger.debug("kl_divergence")
        #posterior: distribution with logits from each hierarchy level/layer
        #posterior_samples: reparameterised output of posterior
        if len(posterior)>1 and self.training: #this posterior has multiple latent layers
            logger.debug("kld for training posterior with more than one latent layer")
            
            logit_list=[]
            samples_last_layer_marginalized=[]
            for i in range(len(posterior)):
                #TODO clip values
                approx_posterior_logit=posterior[i].logits
                logit_list.append(approx_posterior_logit)
            
                if i==len(posterior_samples)-1:
                    approx_posterior_logit_marginalised=torch.sigmoid(approx_posterior_logit)
                else:
                    posterior_sample=posterior_samples[i]
                    #TODO the 0.5 here is a guess. The original code checks if
                    #the posterior samples are greater than 0. It seems my
                    #samples are all positive. Training bias? Mean x?
                    approx_posterior_logit_marginalised=torch.where(posterior_sample>0.5,torch.ones(posterior_sample.size()),torch.zeros(posterior_sample.size()))
                samples_last_layer_marginalized.append(approx_posterior_logit_marginalised)
            logits_concat=torch.cat(logit_list,1)
            #TODO changed treatment wrt to DWave
            # samples_last_layer_marginalized_concat=torch.cat(samples_last_layer_marginalized,1)
            samples_last_layer_marginalized_concat=samples_last_layer_marginalized
            # print(samples_last_layer_marginalized_concat.size())
            kl_div_posterior=self.kl_div_posterior_gradient(approx_post_logits=logits_concat,approx_post_binary_samples=samples_last_layer_marginalized_concat)#DVAE Eq11 - gradient of AE model
            # print(kl_div_posterior)            
            kl_div_prior=self.kl_div_prior_gradient(approx_post_logits=logits_concat,approx_post_binary_samples=samples_last_layer_marginalized_concat)  #DVAE Eq12 - gradient of prior   
            kld=kl_div_prior+kl_div_posterior 
            return kld
        else: # either this posterior only has one latent layer or we are not looking at training
            # #this posterior is not hierarchical - a closed analytical form for the KLD term can be constructed
            # #the mean-field solution (num_latent_hierarchy_levels == 1) reduces to log_ratio = 0.
            # logger.debug("kld for evaluation/training of one layer posterior")
            # entropy=0
            # entropy_reduced=0
            # cross_entropy=0
            # # cross_entropy_reduced=0
            # #TODO implement these functions in distributions!   
            # # for factorial in posterior:
            # for factorial, samples in zip(posterior, posterior_samples):
            #     entropy += factorial.entropy(samples)
            #     # print(entropy.size()) #returns [number samples, number latent layers]
            #     entropy_reduced=torch.sum(entropy,dim=1)
            #     # print(entropy_reduced) # number of samples times a float
            #     #TODO why is this only "samples" in DWave code? Looks like
            #     #they'd only take the last element of the posterior_samples list.
            #     cross_entropy+=self.prior.cross_entropy(samples)
            # return cross_entropy - entropy_reduced
            return 0

    def generate_samples(self, n_samples=100):
        logger.debug("ERROR generate_samples")
        """ It will randomly sample from the model using ancestral sampling. It first generates samples from p(z_0).
        Then, it generates samples from the hierarchical distributions p(z_j|z_{i < j}). Finally, it forms p(x | z_i).  
        
         Args:
             num_samples: an integer value representing the number of samples that will be generated by the model.
        """
        logger.debug("ERROR generate_samples")
        prior_samples = self.prior.get_samples(n_samples)
        # prior_samples = tf.slice(prior_samples, [0, 0], [num_samples, -1])
        
        output_samples = self.decoder.decode_posterior_sample(prior_samples)
        # output_activations[0] = output_activations[0] + self.train_bias
        # output_dist = FactorialBernoulliUtil(output_activations)
        # output_samples = tf.nn.sigmoid(output_dist.logit_mu)
        # print("--- ","end VAE::generate_samples()")
        return output_samples             

    def forward(self, in_data):
        logger.debug("forward")
        #TODO this should yield posterior distribution and samples
        #this now (200806) gives back "smoother" and samples from smoother. Not
        #hierarchical yet.
        posterior_distributions, posterior_samples = self.encoder.hierarchical_posterior(in_data.view(-1, 784))
        posterior_samples_concat=torch.cat(posterior_samples,1)
        
        #take samples z and reconstruct output with decoder
        output_activations = self.decoder.decode(posterior_samples_concat)
        #TODO add bias to output_activations
        # print(output_activations)
        output_distribution = Bernoulli(output_activations)
        output=torch.sigmoid(output_activations)
        return output, output_activations, output_distribution, \
            posterior_distributions, posterior_samples

if __name__=="__main__":
    logger.debug("Testing Model Setup") 
    # model=VAE()
    # model=DiVAE()
    model=HiVAE(latent_dimensions=100)
    print(model.encoder)
    logger.debug("Success")
    pass