"""
Contrastive Divergence sampling routine
"""

import torch
from torch import nn

from models.samplers.baseSampler import BaseSampler
from DiVAE import logging
logger = logging.getLogger(__name__)

class ContrastiveDivergence(BaseSampler):
	def __init__(self, **kwargs):
		super(ContrastiveDivergence, self).__init__(**kwargs)
		
		# #TODO these are the nn.Parameters of the RBM. It's not good design that
		# #these should be members of this class too. But they are required in the
		# #CD training.
		self.rbm_weights = None
		self.rbm_visible_bias = None
		self.rbm_hidden_bias = None

		# NN training parameters adjusted during CD
		self.weights_update = None
		self.visible_bias_update = None
		self.hidden_bias_update = None

	def set_rbm_parameters(self,rbm):
		self.rbm_weights = rbm.get_weights()
		self.rbm_visible_bias = rbm.get_visible_bias()
		self.rbm_hidden_bias = rbm.get_hidden_bias()

		# NN training parameters adjusted during CD
		self.weights_update= torch.zeros(self.rbm_weights.size())
		self.visible_bias_update = torch.zeros(self.rbm_visible_bias.size())
		self.hidden_bias_update =  torch.zeros(self.rbm_hidden_bias.size())

	def run_training(self, input_data):
		reco_data=self._contrastive_divergence(input_data)
		return loss(input_data,reco_data)

	def loss(self, input_data, reco_data):
		# Compute reconstruction error
		loss_fct = torch.nn.MSELoss(reduction='none')
		return loss_fct(input_data, reco_data).sum()

	def sample_from_hidden(self, probabilities_visible):
		output_hidden = torch.matmul(probabilities_visible, self.rbm_weights) + self.rbm_hidden_bias
		probabilities_hidden = torch.sigmoid(output_hidden)
		return probabilities_hidden

	def sample_from_visible(self, probabilities_hidden):
		output_visible = torch.matmul(probabilities_hidden, self.rbm_weights.t()) + self.rbm_visible_bias
		probabilities_visible = torch.sigmoid(output_visible)
		return probabilities_visible

	# Heart of the CD training: Gibbs Sampling
	def gibbs_sampling(self,output_hidden):
		for step in range(self.n_gibbs_sampling_steps):
			probabilities_visible = self.sample_from_visible(output_hidden)
			probabilities_hidden = self.sample_from_hidden(probabilities_visible)
			#When using CDn, only the final update of the hidden nodes should use the probability.
			output_hidden = (probabilities_hidden >= torch.rand(self.n_hidden)).float()
		return probabilities_visible,probabilities_hidden

	def _contrastive_divergence(self, input_data):
		## Estimate the positive phase of the CD
		# start by sampling from the hidden layer using the current input data
		probabilities_hidden_pos = self.sample_from_hidden(input_data)
		# UTML TR 2010–003 says for data driven sampling from the hidden layer
		# the hidden unit turns on if this probability is greater than a random number uniformly distributed 
		# between 0 and 1. THis is stochastically sampled by activating hj only
		# if it's value is greater than a uniformly picked rnd var.
		output_hidden_pos = (probabilities_hidden_pos >= torch.rand(self.n_hidden)).float()
		# this is <pihj>_Data
		associations_pos = torch.matmul(input_data.t(), output_hidden_pos)

		## Estimate the negative phase of the CD
		probabilities_visible_neg,probabilities_hidden_neg=self.gibbs_sampling(output_hidden_pos)
		# this is <pipj>_recon
		associations_neg = torch.matmul(probabilities_visible_neg.t(), probabilities_hidden_neg)

		## Update parameters
		# first iteration, this stays 0
		self.weights_update *= self.momentum_coefficient
		self.weights_update += (associations_pos - associations_neg)
		
		#TODO is this correct? L2 regulariation.
		#https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
		#learning rate should be multiplied to this weight decay update
		#training turns out much more successful!
		self.weights_update -= self.rbm_weights * self.weight_decay_factor  # L2 weight decay

		## simplified version of the same learning rule that uses the states of individual nodes
		self.visible_bias_update *= self.momentum_coefficient
		self.visible_bias_update += torch.sum(input_data - probabilities_visible_neg, dim=0)

		self.hidden_bias_update *= self.momentum_coefficient
		self.hidden_bias_update += torch.sum(probabilities_hidden_pos - probabilities_hidden_neg, dim=0)

		# batch_size = input_data.size(0)

		self.rbm_weights += self.weights_update * self.learning_rate #/ batch_size
		self.rbm_visible_bias += self.visible_bias_update * self.learning_rate #/ batch_size
		self.rbm_hidden_bias += self.hidden_bias_update * self.learning_rate #/ batch_size

		return probabilities_visible_neg
	
	def get_samples(self, n_latent_nodes=100, n_gibbs_sampling_steps=10, sampling_mode="ancestral"):
		logger.debug("generate_samples")
		assert sampling_mode=="gibbs_ancestral" \
			or sampling_mode=="gibbs_flat" \
			or sampling_mode=="random", "Unknown sampling mode"

		#TODO this is only defined for 4 hierarchy layers at the moment. 
		#TODO the range of this is taken from the clamping of the posterior
		#samples to -88,88. Where is this coming from? Check the values again.
		#TODO check if these are proper sampling procedures. Should this be PCD
		#or similar?
		z0=-166*torch.rand([n_latent_nodes])+88
		z1=-166*torch.rand([n_latent_nodes])+88
		z2=-166*torch.rand([n_latent_nodes])+88
		z3=-166*torch.rand([n_latent_nodes])+88

		############
		##Sampling mode: random
		############
		#flat, uniform sampled z, no dependence. Straight to decoder.
		if sampling_mode=="random":
			return [z0,z1,z2,z3]

		############
		##Sampling mode: gibbs_flat
		############
		# start with random z0,z1. Gibbs sampling from trained RBM.
		init_samples_left=torch.cat([z0,z1])
		#this is only used if no gibbs sampling invoked. Remove?
		init_samples_right=torch.cat([z2,z3])

		left=init_samples_left
		right=init_samples_right
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.sample_from_hidden(left)
			left=self.sample_from_visible(right)

		z0_fin,z1_fin = torch.split(left,split_size_or_sections=int(n_latent_nodes))
		z2_fin,z3_fin = torch.split(right,split_size_or_sections=int(n_latent_nodes))

		if sampling_mode=="gibbs_flat":
			return [z0_fin,z1_fin,z2_fin,z3_fin]

		############
		##Sampling mode: gibbs_ancestral
		############
		# start with random z0. Do gibbs sampling. Resulting z0, z1 are new left
		# side of Gibbs Sampling. Repeat until all z sampled in dependence on
		# each other...
		final_sample=[]
		final_sample.append(z0_fin)

		#z1 from uniform random
		init_samples_left=torch.cat([z0_fin,z1])
		init_samples_right=torch.cat([z2,z3])

		left=init_samples_left
		right=init_samples_right
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.sample_from_hidden(left)
			left=self.sample_from_visible(right)

		_,z1_fin=torch.split(left,split_size_or_sections=int(n_latent_nodes))
		final_sample.append(z1_fin)

		init_samples_left=torch.cat([z0_fin,z1_fin])
		init_samples_right=torch.cat([z2,z3])

		left=init_samples_left
		right=init_samples_right
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.sample_from_hidden(left)
			left=self.sample_from_visible(right)

		z2_fin,_=torch.split(right,split_size_or_sections=int(n_latent_nodes))
		final_sample.append(z2_fin)

		init_samples_left=torch.cat([z0_fin,z1_fin])
		init_samples_right=torch.cat([z2_fin,z3])

		left=init_samples_left
		right=init_samples_right
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.sample_from_hidden(left)
			left=self.sample_from_visible(right)

		_,z3_fin=torch.split(right,split_size_or_sections=int(n_latent_nodes))
		final_sample.append(z3_fin)

		return final_sample
	
	def get_samples_kld(self, approx_post_samples=None, n_gibbs_sampling_steps=10):
		# feed data to hidden layer and sample response
		# we feed binarised data sampled from MNIST
		visible=approx_post_samples.detach()
		hidden=None
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			hidden=self.sample_from_hidden(visible)
			visible=self.sample_from_visible(hidden)

		rbm_samples=torch.cat([visible,hidden],dim=1)
		return rbm_samples
	
if __name__=="__main__":
	logger.info("Testing CD Setup")
	sampler=Contrastive_Divergence()
	logger.info("Success")