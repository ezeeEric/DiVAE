"""
PyTorch implementation of a restricted Boltzmann machine

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import numpy as np
import torch

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

class RBM(nn.Module):
	def __init__(self, n_visible, n_hidden, learning_rate=1e-3, n_gibbs_sampling_steps = 1, **kwargs):
		super(RBM, self).__init__(**kwargs)

		r""" A class for an RBM that is co-trained with the rest of a VAE.
		:param int n_visible: Number of visible nodes.
		:param int n_hidden: Number of hidden nodes.
		:param float eps: Learning rate
		"""

		self.n_visible = n_visible
		self.n_hidden = n_hidden

		# Hyperparameters for CDn training
		self.learning_rate = learning_rate
		self.momentum_coefficient = 0.5
		self.n_gibbs_sampling_steps = n_gibbs_sampling_steps
		self.weight_cost = 1e-4        

		# The Sampler (CD) is constructed from this instance of RBM.
		# CD _getattr_ function is modified such that updates to this instance
		# are propagated.
		self.sampler=Contrastive_Divergence(
			n_visible = self.n_visible,
			n_hidden = self.n_hidden,
			learning_rate=self.learning_rate,
			momentum_coefficient=self.momentum_coefficient,
			n_gibbs_sampling_steps=self.n_gibbs_sampling_steps,
			weight_cost=self.weight_cost  
		)
	
	def get_visible_bias(self):
		return self.sampler._visible_bias
	
	def get_hidden_bias(self):
		return self.sampler._hidden_bias
	
	def get_weights(self):
		return self.sampler._weights

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
			right=self.sampler.sample_from_hidden(left)
			left=self.sampler.sample_from_visible(right)

		z0_fin, z1_fin = torch.split(left,split_size_or_sections=int(n_latent_nodes))
		z2_fin, z3_fin = torch.split(right,split_size_or_sections=int(n_latent_nodes))
		
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
			right=self.sampler.sample_from_hidden(left)
			left=self.sampler.sample_from_visible(right)
		
		_,z1_fin=torch.split(left,split_size_or_sections=int(n_latent_nodes))
		final_sample.append(z1_fin)

		init_samples_left=torch.cat([z0_fin,z1_fin])
		init_samples_right=torch.cat([z2,z3])

		left=init_samples_left
		right=init_samples_right
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.sampler.sample_from_hidden(left)
			left=self.sampler.sample_from_visible(right)

		z2_fin,_=torch.split(right,split_size_or_sections=int(n_latent_nodes))
		final_sample.append(z2_fin)

		init_samples_left=torch.cat([z0_fin,z1_fin])
		init_samples_right=torch.cat([z2_fin,z3])

		left=init_samples_left
		right=init_samples_right
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			right=self.sampler.sample_from_hidden(left)
			left=self.sampler.sample_from_visible(right)

		_,z3_fin=torch.split(right,split_size_or_sections=int(n_latent_nodes))
		final_sample.append(z3_fin)
		
		return final_sample
	
	def get_samples_kld(self, approx_post_samples=None, n_gibbs_sampling_steps=10):
		logger.debug("generate_samples")
		# feed data to hidden layer and sample response
		# we feed binarised data sampled from MNIST
		visible=approx_post_samples.detach()
		hidden=None
		for gibbs_step in range(0,n_gibbs_sampling_steps):
			hidden=self.sampler.sample_from_hidden(visible)
			visible=self.sampler.sample_from_visible(hidden)

		rbm_samples=torch.cat([visible,hidden],dim=1)
		return rbm_samples

	def train_sampler(self, in_data):
		""" Use sampler to train rbm. dData is the current batch."""
		loss=0
		for img in in_data:
			loss+=self.sampler.contrastive_divergence(img)
		return loss
	
	def full_training(self, in_data, epoch):
		""" Use sampler to train rbm. Data is the current batch."""
		loss = 0    
		loss=self.sampler.contrastive_divergence_fixed_hid_vis(in_data)
		logger.info('Epoch {0}. Loss={1:.2f}'.format(epoch,loss))
		return
			
	def get_logZ_value(self):
		logger.debug("ERROR get_logZ_value")
		#TODO Figure 10 Rolfe
		return 33.65
	
	def energy(self, post_samples):
		"""Energy Computation for RBM
		vis*b_vis+hid*b_hid+vis*w*hid
		Takes posterior samples as input
		"""
		post_samples_concat=torch.cat(post_samples,dim=1)
		n_split=post_samples_concat.size()[1]//2
		post_samples_left,post_samples_right=torch.split(post_samples_concat,split_size_or_sections=int(n_split),dim=1)

		v=post_samples_left
		h=post_samples_right

		e_vis=torch.matmul(v,self.get_visible_bias())
		e_hid=torch.matmul(h,self.get_hidden_bias())
		e_mix=torch.sum(torch.matmul(v,self.get_weights())*h,axis=1)
		energy=-e_vis-e_hid-e_mix
		return energy

	def cross_entropy(self,post_samples):
		#TODO currently (200827) returns exactly the energy, as logZ=0
		"""For the case in Discrete VAE paper, this is simply the log prob (verify!)"""
		return -self.log_prob(post_samples)

	def log_prob(self, samples):
		"""log(exp(-E(z))/Z) = -E(z) - log(Z)"""
		return -self.energy(samples)-self.get_logZ_value()