"""
CDn Gibbs Sampler

Implements Gibbs Sampling/CDn procedure as described in UTML TR 2010–003.
This is an approximation to the original Contrastive Divergence algorithm.
"""

import torch
from torch import nn

from models.samplers.baseSampler import BaseSampler
from DiVAE import logging
logger = logging.getLogger(__name__)

class CDnGibbsSampler(BaseSampler):
	def __init__(self, RBM, **kwargs):
		super(CDnGibbsSampler, self).__init__(**kwargs)
		
		self._RBM = RBM

		# NN training parameters adjusted during CD
        self.weights_update = torch.zeros(self._RBM.get_weights().size())
        self.visible_bias_update = torch.zeros(self.rbm_visible_bias.size())
        self.hidden_bias_update = torch.zeros(self.rbm_hidden_bias.size())
        
	def hidden_samples(self, probabilities_visible):
		output_hidden = torch.matmul(probabilities_visible, self._RBM.get_weights()) + self.rbm_hidden_bias
		probabilities_hidden = torch.sigmoid(output_hidden)
		return probabilities_hidden

	def visible_samples(self, probabilities_hidden):
		output_visible = torch.matmul(probabilities_hidden, self._RBM.get_weights().t()) + self.rbm_visible_bias
		probabilities_visible = torch.sigmoid(output_visible)
		return probabilities_visible

	# Heart of the CDn training: Alternating Gibbs Sampling
	def gibbs_sampling(self,input_sample, n_gibbs_sampling_steps):
		# feed data to hidden layer and sample response
		left=input_sample
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.hidden_samples(left)
			left=self.visible_samples(right)
		return left, right

		# for step in range(self.n_gibbs_sampling_steps):
		# 	probabilities_visible = self.visible_samples(output_hidden)
		# 	probabilities_hidden = self.hidden_samples(probabilities_visible)
		# 	#When using CDn, only the final update of the hidden nodes should use the probability.
		# 	output_hidden = (probabilities_hidden >= torch.rand(self.n_hidden)).float()
		# return probabilities_visible,probabilities_hidden

	#this method is used to generate samples 
	def get_samples(self, approx_post_samples=[], n_latent_nodes=100, n_latent_hierarchy_lvls=4, n_gibbs_sampling_steps=10):
		logger.debug("get_samples")
		##Sampling mode: gibbs 
		# start with random numbers left side rbm. Gibbs sampling from
		# trained RBM.
		assert len(n_latent_hierarchy_lvls)%2==0, "Number of hierarchy layers should be even")
		
		#these are the starting samples for the gibbs sampling - the left
		#(visible) side of the rbm. If no sample is given, random numbers are
		#picked.
		if len(approx_post_samples)<1:
			#TODO check if this is doing the right ting
			for i in range(0,n_latent_hierarchy_lvls//2):
				#TODO the range of this is taken from the clamping of the posterior
				#samples to -88,88. Where is this coming from? Check the values again.
				approx_post_samples.append(-166*torch.rand([n_latent_nodes])+88)

		#TODO should this be detached?
		left=approx_post_samples.detach()
		left,right=self.gibbs_sampling(left.detach(),n_gibbs_sampling_steps)
		#TODO see below
		rbm_samples=[left,right]
		return rbm_samples
		#TODO check where this is returned to. Do concatenation there.
		# z0_fin,z1_fin = torch.split(left,split_size_or_sections=int(n_latent_nodes))
		# z2_fin,z3_fin = torch.split(right,split_size_or_sections=int(n_latent_nodes))
		# return [z0_fin,z1_fin,z2_fin,z3_fin]
		# return torch.cat([left,right],dim=1)
	
######################
# The following methods are for standalone RBM training and not used in the
# DiVAE models
######################

	def run_training(self, input_data,n_gibbs_sampling_steps,learning_rate,momentum_coefficient,weight_decay_factor):
		reco_data=self._cdn_gibbs_sampling(input_data,n_gibbs_sampling_steps,learning_rate,momentum_coefficient,weight_decay_factor)
		return loss(input_data,reco_data)

	def loss(self, input_data, reco_data):
		# Compute reconstruction error
		loss_fct = torch.nn.MSELoss(reduction='none')
		return loss_fct(input_data, reco_data).sum()

	def _cdn_gibbs_sampling(self, input_data,n_gibbs_sampling_steps,learning_rate,momentum_coefficient,weight_decay_factor):
		## Estimate the positive phase of the CD
		# start by sampling from the hidden layer using the current input data
		probabilities_hidden_pos = self.hidden_samples(input_data)
		# UTML TR 2010–003 says for data driven sampling from the hidden layer
		# the hidden unit turns on if this probability is greater than a random number uniformly distributed 
		# between 0 and 1. THis is stochastically sampled by activating hj only
		# if it's value is greater than a uniformly picked rnd var.
		output_hidden_pos = (probabilities_hidden_pos >= torch.rand(self.n_hidden)).float()
		# this is <pihj>_Data
		associations_pos = torch.matmul(input_data.t(), output_hidden_pos)

		## Estimate the negative phase of the CD
		output_hidden=output_hidden_pos
		for step in range(n_gibbs_sampling_steps):
			probabilities_visible = self.visible_samples(output_hidden)
			probabilities_hidden = self.hidden_samples(probabilities_visible)
			#When using CDn, only the final update of the hidden nodes should use the probability.
			output_hidden = (probabilities_hidden >= torch.rand(self.n_hidden)).float()
		probabilities_visible_neg=probabilities_visible
		probabilities_hidden_neg=probabilities_hidden
		
		# this is <pipj>_recon
		associations_neg = torch.matmul(probabilities_visible_neg.t(), probabilities_hidden_neg)

		## Update parameters
		# first iteration, this stays 0
		self.weights_update *= momentum_coefficient
		self.weights_update += (associations_pos - associations_neg)
		
		#TODO is this correct? L2 regulariation.
		#https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
		#learning rate should be multiplied to this weight decay update
		#training turns out much more successful!
		self.weights_update -= self._RBM.get_weights() * weight_decay_factor  # L2 weight decay

		## simplified version of the same learning rule that uses the states of individual nodes
		self.visible_bias_update *= momentum_coefficient
		self.visible_bias_update += torch.sum(input_data - probabilities_visible_neg, dim=0)

		self.hidden_bias_update *= momentum_coefficient
		self.hidden_bias_update += torch.sum(probabilities_hidden_pos - probabilities_hidden_neg, dim=0)

		# batch_size = input_data.size(0)

		self._RBM.get_weights() += self.weights_update * learning_rate #/ batch_size
		self.rbm_visible_bias += self.visible_bias_update * learning_rate #/ batch_size
		self.rbm_hidden_bias += self.hidden_bias_update * learning_rate #/ batch_size

		return probabilities_visible_neg
	
if __name__=="__main__":
	logger.info("Testing CD Setup")
	sampler=Contrastive_Divergence()
	logger.info("Success")