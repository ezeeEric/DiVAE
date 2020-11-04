# -*- coding: utf-8 -*-
"""
PyTorch implementation of a restricted Boltzmann machine

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import numpy as np
import torch

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

import logging
logger = logging.getLogger(__name__)

#Gibbs Sampler, manual implementation
class Contrastive_Divergence(object):
	def __init__(self, n_visible, n_hidden, learning_rate,momentum_coefficient,n_gibbs_sampling_steps,weight_cost,**kwargs):
		
		# Hyperparameters for CDn training
		#TODO duplicated between RBM and CD classes - find better solution
		self.learning_rate = learning_rate
		self.momentum_coefficient = momentum_coefficient
		self.n_gibbs_sampling_steps = n_gibbs_sampling_steps
		self.weight_cost = weight_cost 
		
		self.n_visible=n_visible #from rbm
		self.n_hidden=n_hidden #from rbm

		# NN training parameters adjusted during CD
		self.weights_update= torch.zeros(n_visible, n_hidden)
		self.visible_bias_update = torch.zeros(n_visible)
		self.hidden_bias_update = torch.zeros(n_hidden)

		# random weights and biases for all layers
		# weights between visible and hidden nodes. 784x128 (that is 28x28 input
		#size, 128 arbitrary choice)
		#arbitrarily scaled by 0.01 
		self._weights = torch.randn(n_visible, n_hidden) * 0.01
  		#all biases initialised to 0.5
		self._visible_bias = torch.ones(n_visible) * 0.5
		#applying a 0 bias to the visible node
		self._hidden_bias = torch.zeros(n_hidden)
	
	# #handmade inheritance
	# def __getattr__(self,name):
	# 	if name in self.__dict__:
	# 		return self.__dict__[name]
	# 	elif name in self.rbm.__dict__:
	# 		return self.rbm.__dict__[name]
	# 	else: 
	# 		raise AttributeError

	def sample_from_hidden(self, probabilities_visible):
		output_hidden = torch.matmul(probabilities_visible, self._weights) + self._hidden_bias
		probabilities_hidden = torch.sigmoid(output_hidden)
		return probabilities_hidden

	def sample_from_visible(self, probabilities_hidden):
		output_visible = torch.matmul(probabilities_hidden, self._weights.t()) + self._visible_bias
		probabilities_visible = torch.sigmoid(output_visible)
		return probabilities_visible
	# Heart of the CD training: Gibbs Sampling
	#  	
	def gibbs_sampling(self,output_hidden):
		
		for step in range(self.n_gibbs_sampling_steps):
			probabilities_visible = self.sample_from_visible(output_hidden)
			probabilities_hidden = self.sample_from_hidden(probabilities_visible)
			#When using CDn, only the final update of the hidden units should use the probability.
			output_hidden = (probabilities_hidden >= torch.rand(self.n_hidden)).float()
		return probabilities_visible,probabilities_hidden

	def contrastive_divergence(self, input_data):

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
		self.weights_update -= self._weights * self.weight_cost  # L2 weight decay

		## simplified version of the same learning rule that uses the states of individual units
		self.visible_bias_update *= self.momentum_coefficient
		self.visible_bias_update += torch.sum(input_data - probabilities_visible_neg, dim=0)

		self.hidden_bias_update *= self.momentum_coefficient
		self.hidden_bias_update += torch.sum(probabilities_hidden_pos - probabilities_hidden_neg, dim=0)

		# batch_size = input_data.size(0)

		self._weights += self.weights_update * self.learning_rate #/ batch_size
		self._visible_bias += self.visible_bias_update * self.learning_rate #/ batch_size
		self._hidden_bias += self.hidden_bias_update * self.learning_rate #/ batch_size

		# Compute reconstruction error
		#loss = torch.sum((input_data - probabilities_visible_neg)**2)
		loss_fct = torch.nn.MSELoss(reduction='none')
		loss = loss_fct(input_data, probabilities_visible_neg).sum()
		return loss

class RBM(Distribution):
	def __init__(self, n_visible, n_hidden, learning_rate=1e-3, n_gibbs_sampling_steps = 5, **kwargs):
		super(Distribution, self).__init__(**kwargs)

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

	def get_samples(self, samples, random=False):
		logger.info("generate_samples")
		if random:
			samples=torch.rand(samples.size())
		# feed data to hidden layer and sample response
		# we feed binarised data sampled from MNIST
		hidden=self.sampler.sample_from_hidden(samples)
		visible=self.sampler.sample_from_visible(hidden)
		return visible

	def train(self, data):
		""" Use sampler to train rbm. Data is the current batch."""
		loss=0
		for x in data:
			img=x.view(1, x.size()[0])
			loss+=self.sampler.contrastive_divergence(img)
		return loss

	def get_logZ_value(self):
		logger.debug("ERROR get_logZ_value")
		return 0
	
	def __repr__(self):
		return "RBM"

	def energy(self, post_samples):
		"""Energy Computation for RBM
		vis*b_vis+hid*b_hid+vis*w*hid
		Takes posterior samples as input
		"""
		#v=post_samples
		#TODO is this randomised enough when used in VAE? Practical guide
		#recommends sampling... (Sec 3.4)
		e_vis=torch.matmul(post_samples,self.get_visible_bias())
		#h
		h=self.sampler.sample_from_hidden(post_samples)
		e_hid=torch.matmul(h,self.get_hidden_bias())
		e_mix=torch.sum(torch.matmul(post_samples,self.get_weights())*h,axis=1)
		energy=-e_vis-e_hid-e_mix
		return energy

	def cross_entropy(self,post_samples):
		#TODO currently (200827) returns exactly the energy, as logZ=0
		"""For the case in Discrete VAE paper, this is simply the log prob (verify!)"""
		return -self.log_prob(post_samples)

	def log_prob(self, samples):
		"""log(exp(-E(z))/Z) = -E(z) - log(Z)"""
		logger.debug("log_prob")
		return -self.energy(samples)-self.get_logZ_value()

if __name__=="__main__":
	logger.info("Testing RBM Setup")

	BATCH_SIZE = 32
	VISIBLE_UNITS = 784  # 28 x 28 images
	HIDDEN_UNITS = 128
	N_GIBBS_SAMPLING_STEPS = 10
	EPOCHS = 1
	N_EVENTS_TRAIN=100
	N_EVENTS_TEST=-1
	do_train=True
	config_string="_".join(map(str,[N_EVENTS_TRAIN,EPOCHS,N_GIBBS_SAMPLING_STEPS]))

	from data.loadMNIST import loadMNIST
	train_loader,test_loader=loadMNIST(
			batch_size=BATCH_SIZE,
			num_evts_train=N_EVENTS_TRAIN,
			num_evts_test=N_EVENTS_TEST,
			binarise="threshold")
	
	rbm = RBM(n_visible=VISIBLE_UNITS, n_hidden=HIDDEN_UNITS, n_gibbs_sampling_steps=N_GIBBS_SAMPLING_STEPS)

	if do_train:
		for epoch in range(EPOCHS):
			loss = 0    
			for batch_idx, (x_true, label) in enumerate(train_loader):
				loss_per_batch = rbm.train(x_true.view(-1,VISIBLE_UNITS))
				loss += loss_per_batch
			loss /= len(train_loader.dataset)
			logger.info('Epoch {0}. Loss={1:.2f}'.format(epoch,loss))
		import os
		logger.info("Saving Model")
		f=open("./output/rbm_test_200827_wdecay_{0}.pt".format(config_string),'wb')
		torch.save(rbm.sampler,f)
		f.close()
	else:
		f=open("./output/rbm_test_200827_wdecay_{0}.pt".format(config_string),'rb')
		rbm.sampler=torch.load(f)
		f.close()
	
	# # ########## EXTRACT FEATURES ##########
	logger.info("Sampling from RBM")
	for batch_idx, (x_true, label) in enumerate(test_loader):
		print(x_true.size())
		print(label)
		y=rbm.get_samples(x_true.view(-1,VISIBLE_UNITS))
		energy=rbm.energy(x_true.view(-1,VISIBLE_UNITS))
		print(energy)
		cross_entropy=rbm.cross_entropy(x_true.view(-1,VISIBLE_UNITS))
		print(cross_entropy)
		# use a random picture for sanity checks
		# samples=torch.rand(x_true.view(-1,VISIBLE_UNITS).size())
		# yrnd=rbm.get_samples(x_true.view(-1,VISIBLE_UNITS), random=True)
		# energy=rbm.energy(samples)
		# print(energy)
		break
	print(y.size())
	from helpers import plot_MNIST_output

	plot_MNIST_output(x_true,y, n_samples=5, output="./output/rbm_test_200827_wdecay_{0}.png".format(config_string))
	# plot_MNIST_output(x_true,yrnd, n_samples=5, output="./output/rbm_test_200827_rnd_{0}.png".format(config_string))

	# train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
	# train_labels = np.zeros(len(train_dataset))
	# test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
	# test_labels = np.zeros(len(test_dataset))
	
	# for i, (batch, labels) in enumerate(train_loader):
	#     batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
	#     train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_from_hidden(batch).cpu().numpy()
	#     train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()
	
	# for i, (batch, labels) in enumerate(test_loader):
	#     batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
	#     test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_from_hidden(batch).cpu().numpy()
	#     test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()