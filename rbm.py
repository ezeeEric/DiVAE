# -*- coding: utf-8 -*-
"""
PyTorch implementation of a restricted Boltzmann machine
Based on: http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

and the Wikipedia article about RBMs.

Author: Eric Drechsler (eric_drechsler@sfu.ca)

Based on work from Olivia di Matteo.

"""

import numpy as np
import torch

# from networks import Prior

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

# Keep things simple for now
class RBM(Distribution):
	def __init__(self, n_visible, n_hidden, eps=1e-3, **kwargs):
		super(Distribution, self).__init__(**kwargs)

		r""" A class for an RBM that is co-trained with the rest of a VAE.
		:param int n_visible: Number of visible nodes.
		:param int n_hidden: Number of hidden nodes.
		:param float eps: Learning rate
		"""

		self.n_visible = n_visible
		self.n_hidden = n_hidden

		# Learning rate
		self._eps = eps

		# Initialize random values for the weights and biases of all layers
		# According to Hinton, use small normally distributed values for the weights
		self._weights = nn.Parameter(Normal(loc=0, scale=0.01).sample((n_hidden, n_visible)))
		
		# Should use the proportion of training vectors in which unit i is turned on
		# For now let's just set them randomly 
		self._visible_bias = nn.Parameter(Uniform(low=0, high=1).sample((n_visible, )))

		# Unless there is some sparsity, initialize these all to 0
		self._hidden_bias = nn.Parameter(torch.zeros((n_hidden, )))
	
	def get_samples(self,n_samples):
		logger.error("generate_samples")
		#TODO is this correct? we sample from a uniform distribution, the
		#sampled probabilities are then fe to the hidden layer of the RBM, to
		#produce the bvisible nodes it has learned.
		random_probabilities=torch.rand((n_samples,n_visible))
		# sample from visible
		visible=random_probabilities 
		return visible

	def energy(self, v, h):
		# Pass a configuration of visible and hidden units to get its energy
		# In principle we could pass 
		visible_e = torch.einsum("...i,i->...", v, self._visible_bias)
		hidden_e = torch.einsum("...i,i->...", v, self._hidden_bias)
		cross_term = torch.einsum("...i,ij,...j->...", v, self._weights, h)

		return -(visible_e + hidden_e + cross_term)

	def hidden_energy(self, z):
		bias_term = torch.einsum("...i,i->...", z, self._hidden_bias)
		weight_term = torch.einsum("...i,...i->...", z ,torch.einsum("ij,...i->...i", self._weights, z))
		return - bias_term - weight_term 

	def visible_to_hidden(self, v):
		activations = torch.einsum("ij,j->i", self._weights, v) + self._hidden_bias
		probs = torch.sigmoid(activations)
		return self.sample(probs)

	def hidden_to_visible(self, h):
		activations = torch.einsum("ij,j->i", self._weights, h)
		probs = torch.sigmoid(activations)
		return self.sample(probs)

	def sample(self, probs):
		# Treat the probabilities like a Bernoulli distribution
		bernoulli_probs = torch.distributions.Bernoulli(probs=probs)
		return bernoulli_probs.sample((1, ))[0]

	def gibbs_sample(self, v, n_samples=10):
		for idx in range(n_samples):
			h = self.visible_to_hidden(v)
			v = self.hidden_to_visible(h)
		return v

	def contrastive_divergence(self, v):
		# From the Wiki
		# 1) Compute h from h
		h = self.visible_to_hidden(v)

		# 2) Calculate positive gradient
		pos_grad = torch.einsum('i,j->ij', h, v)

		# 3a) Go backwards
		v_prime = self.gibbs_sample(self.hidden_to_visible(h), n_samples=40)

		# 3b) Now sample another h from that
		h_prime = self.visible_to_hidden(v_prime)

		# 4) Calculate the negative gradient
		neg_grad = torch.einsum('i,j->ij', h_prime, v_prime)

		# 5) Update the weight matrix by computing the gradient
		weight_change = self._eps * (pos_grad - neg_grad)
		self._weights += weight_change[0]

		# 6) Update the biases too
		visible_bias_change = self._eps * (v - v_prime)
		self._visible_bias += visible_bias_change[0]

		hidden_bias_change = self._eps * (h - h_prime)
		self._hidden_bias += hidden_bias_change[0]

	def train(self, X):
		""" Train the RBM over the provided batch of data.
		X is a tensor with shape (batch_size, n_visible)
		"""
		#TODO 
		pass
		# for datum in X:
			# self.contrastive_divergence(datum[:self.n_visible])
			#self.contrastive_divergence(datum.view(1, datum.size()[0]))

# @register_kl	
# def kl_divergence(p,q):
# 	pass
	def __repr__(self):
		return "DIY RBM"

if __name__=="__main__":
    print("Testing RBM Setup")
    prior=RBM(4,4)
    print("Success")
    pass