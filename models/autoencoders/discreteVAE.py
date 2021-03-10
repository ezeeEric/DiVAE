"""
Discrete Variational Autoencoder Class

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
from torch import nn
from models.autoencoders.autoencoderbase import AutoEncoderBase

from models.networks.basicCoders import BasicDecoder
from models.networks.hierarchicalEncoder import HierarchicalEncoder
from models.rbm.rbm import RBM
from models.rbm.samplers.contrastiveDivergence import ContrastiveDivergence

from utils.distributions import Bernoulli

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)


class DiVAE(AutoEncoderBase):
	def __init__(self, **kwargs):
		super(DiVAE, self).__init__(**kwargs)
		self._model_type="DiVAE"
		
		#TODO can this be done through inheritance from AutoEncoder?
		self._decoder_nodes=[]
		
		dec_node_list=[(int(self._latent_dimensions*self._config.model.n_latent_hierarchy_lvls))]+list(self._config.model.decoder_hidden_nodes)+[self._flat_input_size]

		for num_nodes in range(0,len(dec_node_list)-1):
			nodepair=(dec_node_list[num_nodes],dec_node_list[num_nodes+1])
			self._decoder_nodes.append(nodepair)

		#TODO change names globally
		#TODO one wd factor for both SimpleDecoder and encoder
		self.weight_decay_factor=self._config.engine.weight_decay_factor
		
        # TODO - Model attributes can be directly imported from the config model dict
        # by iterating over and using (k,v) in dict.items() and using setattr(self, k, v)

		#ENCODER SPECIFICS
		#number of hierarchy levels in encoder. At each hierarchy level an latent layer is formed.
		self.n_latent_hierarchy_lvls=self._config.model.n_latent_hierarchy_lvls

		#number of latent nodes in the prior - output nodes for each level of
		#the hierarchy.
		self.n_latent_nodes=self._config.model.n_latent_nodes
		
		# number of layers in encoder before latent layer. These layers map
		#input to the latent layer. 
		self.n_encoder_layers=self._config.model.n_encoder_layers

		#each hierarchy has NN with n_encoder_layers_enc layers
		#number of deterministic nodes in each encoding layer. 
		self.n_encoder_layer_nodes=self._config.model.n_encoder_layer_nodes

		#added to output activation of last decoder layer in forward call
		self._train_bias=self.set_train_bias()

	def set_train_bias(self):
		"""
		this treatment is a recommendation from the Hinton paper (A Practical Guide to Training Restricted Boltzmann Machines) on how to
		initialise biases. 
		It is usually helpful to initialize the bias of visible unit i to
		log[pi/(1−pi)] 
		where pi is the proportion of training vectors in which unit i is on. If this is not done, 
		the early stage of learning will use the hidden units to make i turn on with a probability of approximately pi.

		Returns:
			
		"""
		clipped_mean=torch.clamp(self._dataset_mean,0.001,0.999).detach()
		return -torch.log(1/clipped_mean-1)

	def create_networks(self):
		logger.debug("Creating Network Structures")
		self.encoder=self._create_encoder()
		self.prior=self._create_prior()
		self.decoder=self._create_decoder()
		return
	
	def _create_encoder(self):
		logger.debug("ERROR _create_encoder dummy implementation")
		return HierarchicalEncoder(
			input_dimension=self._flat_input_size,
			n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
			n_latent_nodes=self.n_latent_nodes,
			n_encoder_layer_nodes=self.n_encoder_layer_nodes,
			n_encoder_layers=self.n_encoder_layers,
			skip_latent_layer=False,
			cfg=self._config)

	def _create_decoder(self):
		logger.debug("_create_decoder")
		#Identity output_activation_fct, as this sigmoid is called manually in forward()
		return BasicDecoder(node_sequence=self._decoder_nodes, activation_fct=self._activation_fct, output_activation_fct=nn.Identity(), cfg=self._config)

	def _create_prior(self):
		logger.debug("_create_prior")
		num_rbm_nodes_per_layer=self._config.model.n_latent_hierarchy_lvls*self._latent_dimensions//2

		rbm=RBM(n_visible=num_rbm_nodes_per_layer,n_hidden=num_rbm_nodes_per_layer)
		rbm.sampler=ContrastiveDivergence( 
			learning_rate=self._config.engine.learning_rate,
			momentum_coefficient=self._config.engine.momentum_coefficient,
			n_gibbs_sampling_steps=self._config.engine.n_gibbs_sampling_steps,
			weight_decay_factor=self._config.engine.weight_decay_factor
		)
		
		exit()
		return rbm
   
	def weight_decay_loss(self):
		#TODO Implement weight decay
		logger.debug("ERROR weight_decay_loss NOT IMPLEMENTED")
		return NotImplementedError

	def loss(self, input_data, fwd_out):
		logger.debug("loss")

		#1) Gradients of KL Divergence     
        kl_loss_per_sample=self.kl_divergence(fwd_out.posterior_distributions,fwd_out.posterior_samples)

        # Bug 1 - KL loss per sample returns an int of 0
        try:
            kl_loss = torch.mean(kl_loss_per_sample)
        except:
            kl_loss = torch.mean(torch.tensor(float(kl_loss_per_sample)))

		#2) AE loss
		ae_loss_matrix=-fwd_out.output_distribution.log_prob_per_var(input_data.view(-1, self._flat_input_size))
		#loss is the sum of all variables (pixels) per sample (event in batch)
        ae_loss_per_sample = torch.sum(ae_loss_matrix,1)
        ae_loss = torch.mean(ae_loss_per_sample)

		#3) weight decay loss
		#TODO add this for encoder, decoder, prior

		#4) final loss
		neg_elbo = ae_loss + kl_loss

		#TODO include the weight decay regularisation in the loss to penalise
		#complexity
		loss=neg_elbo#+weight_decay_loss  
		return {"loss":loss, "ae_loss":ae_loss, "kl_loss":kl_loss}

	def kl_div_prior_gradient(self, posterior_logits, posterior_binary_samples):
		logger.debug("kl_div_prior_gradient")

		#see quadrant implementation
		#DVAE Eq11 - gradient of prior
		#gradient of the KLD between posterior and prior wrt to prior
		#parameters theta, i.e. generative model parameters.

		# Ep(z,theta) = -zT*Weights*z - zT*bias

		######
		# POSITIVE: samples z_i from posterior
		####
		#logits to probabilities
		posterior_probs=torch.sigmoid(posterior_logits)
		positive_probs=posterior_probs.detach()
		
		#samples from posterior are labelled positive
		positive_samples=posterior_binary_samples.detach()

		n_split=positive_samples.size()[1]//2
		positive_samples_left,positive_samples_right=torch.split(positive_samples,split_size_or_sections=int(n_split),dim=1)
		
		#-zT*Weights*z 
		pos_first_term=torch.matmul(positive_samples_left,self.prior.get_weights())*positive_samples_right
	   
		rbm_bias_left=self.prior.get_visible_bias()
		rbm_bias_right=self.prior.get_hidden_bias()
		rbm_bias=torch.cat([rbm_bias_left,rbm_bias_right])
		
		#zT*bias
		#TODO is it correct to use probabilities here? shouldn't these be the
		#binary samples z?
		pos_sec_term=positive_probs*rbm_bias
		
		#positive d/dtheta Ep(z,theta) (z from posterior q, first term in Eq11)
		pos_kld_per_sample=-(torch.sum(pos_first_term,axis=1)+torch.sum(pos_sec_term,axis=1))
		
		######
		# NEGATIVE: samples z_i from prior (RBM)
		####

		#TODO what's the impact of doing gibbs sampling here? Is this the
		#best possible sampling procedure?
		rbm_samples=self.prior.get_samples_kld(approx_post_samples=positive_samples_left,n_gibbs_sampling_steps=1)
		negative_samples=rbm_samples.detach()
		
		# Ep(z,theta) = -zT*Weights*z - zT*bias
		n_split=negative_samples.size()[1]//2
		negative_samples_left,negative_samples_right=torch.split(negative_samples,split_size_or_sections=int(n_split),dim=1)
		neg_first_term=torch.matmul(negative_samples_left,self.prior.get_weights())*negative_samples_right
		
		#TODO in the positive case we use probabilities, in the negative case
		#here samples from the prior... is this correct? Relates to the TODO above.
		neg_sec_term=negative_samples*rbm_bias
		neg_kld_per_sample=(torch.sum(neg_first_term,axis=1)+torch.sum(neg_sec_term,axis=1))
		
		kld_per_sample=pos_kld_per_sample+neg_kld_per_sample
		return kld_per_sample

	def kl_div_posterior_gradient(self, posterior_logits, posterior_binary_samples):
		#see quadrant implementation
		#DVAE Eq12
		#gradient of the KLD between posterior and prior wrt to posterior
		#parameters phi.
		logger.debug("kl_div_posterior_gradient")

		#logits to probabilities. dq/dphi in Eq12. The only differentiated component
		#in this calculation!
		posterior_upper_bound = 0.999*torch.ones_like(posterior_logits)
		posterior_probs=torch.min(posterior_upper_bound, torch.sigmoid(posterior_logits))
		
		#binarised/discretised samples from posterior to RBM layers
		n_split=int(posterior_binary_samples.size()[1]//2)
		rbm_samples_left,rbm_samples_right=torch.split(posterior_binary_samples,split_size_or_sections=n_split,dim=1)

		#the following prepares the variables in the calculation into a specific
		#format, so it's easier to read later.
		rbm_bias_left=self.prior.get_visible_bias()
		rbm_bias_right=self.prior.get_hidden_bias()

		rbm_bias=torch.cat([rbm_bias_left,rbm_bias_right])
		rbm_weight=self.prior.get_weights()

		# this is transposed, so we multiply what we call "right hand" ("hidden layer")
		# samples with right rbm nodes
		rbm_activation_right=torch.matmul(rbm_samples_right,rbm_weight.t())
		rbm_activation_left=torch.matmul(rbm_samples_left,rbm_weight)

		#corresponds to zT*W
		rbm_activation=torch.cat([rbm_activation_right,rbm_activation_left],1)
		
		#Eq 12, (1-z)/(1-q). 
		hadamard_scaling= (1.0 - posterior_binary_samples) / (1.0 - posterior_probs)
		hadamard_scaling_left,hadamard_scaling_right=torch.split(hadamard_scaling, split_size_or_sections=int(n_split),dim=1)
		
		#TODO check again if this is actually true
		#In order to be able to put the differentiated component dq/dphi outside
		#the bracketes, the above is transformed:
		hadamard_scaling_with_ones=torch.cat([hadamard_scaling_left,torch.ones(hadamard_scaling_right.size())],axis=1)
		
		#this gives all terms of Eq12 except dq/dphi
		with torch.no_grad():
			undifferentiated_component=posterior_logits-rbm_bias-rbm_activation*hadamard_scaling_with_ones
			undifferentiated_component=undifferentiated_component.detach()
		
		#final KLD gradient wrt phi
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
		
				#TODO this step is not clear anymore: where was this motivated
				#in the paper? 
				if lvl==len(posterior_distribution)-1:
					samples_marginalised.append(torch.sigmoid(logits))
				else:
					zero_mask=torch.zeros(current_post_samples.size())
					one_mask=torch.ones(current_post_samples.size())
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

	#TODO experimental for now. The sampling technique in the prior is not
	#cross checked with anything.
	def generate_samples(self):
		prior_samples=[]
		#how many samples (i.e. digits) to look at
		for i in range(0,self._config.n_generate_samples):
			prior_sample = self.prior.get_samples(
				n_latent_nodes=self.n_latent_nodes,
				n_gibbs_sampling_steps=self._config.engine.n_gibbs_sampling_steps, 
				sampling_mode=self._config.engine.sampling_mode)
			prior_sample = torch.cat(prior_sample)
			prior_samples.append(prior_sample)
		
		prior_samples=torch.stack(prior_samples)
		output_activations = self.decoder.decode(prior_samples)
		output_activations = output_activations+self._train_bias
		output_distribution = Bernoulli(logit=output_activations)
		output=torch.sigmoid(output_distribution.logits)
		output.detach()

		return output

	def forward(self, input_data):
		logger.debug("forward")

		#see definition for explanation
		out=self._output_container.clear()

		#TODO data prep - study if this does good things
		input_data_centered=input_data.view(-1, self._flat_input_size)-self._dataset_mean
		#Step 1: Feed data through encoder 
		out.posterior_distributions, out.posterior_samples = self.encoder.hierarchical_posterior(input_data_centered)
		posterior_samples_concat=torch.cat(out.posterior_samples,1)
		#Step 2: take samples zeta and reconstruct output with decoder
		output_activations = self.decoder.decode(posterior_samples_concat)

		out.output_activations = output_activations+self._train_bias
		out.output_distribution = Bernoulli(logit=out.output_activations)
		out.output_data = torch.sigmoid(out.output_distribution.logits)
		
		return out

if __name__=="__main__":
	logger.debug("Testing Model Setup") 
	model=DiVAE()
	print(model)
	logger.debug("Success")
