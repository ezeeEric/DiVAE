"""
PyTorch implementation of a restricted Boltzmann machine

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import numpy as np
import torch

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

from DiVAE import logging
logger = logging.getLogger(__name__)

class RBM(nn.Module):
	def __init__(self, n_visible, n_hidden, **kwargs):
		super(RBM, self).__init__(**kwargs)

		self._n_visible=n_visible
		self._n_hidden=n_hidden
 
	    # random weights and biases for all layers
		# weights between visible and hidden nodes. 784x128 (that is 28x28 input
		#size, 128 arbitrary choice)
		# if requires_grad=False : we calculate the weight update ourselves, not
		# through backpropagation
		require_grad=True
		"""
		num_params = int(((n_visible)*(n_visible-1))/2)
		self._weights = nn.Parameter(torch.randn(num_params) * 0.01, requires_grad=require_grad)
		self._W_idxs = torch.triu_indices(n_visible, n_hidden, offset=1)
        	"""
		self._weights = nn.Parameter(torch.randn(n_visible, n_hidden)*0.01, requires_grad=require_grad)
  		# #all biases initialised to 0.5
		self._visible_bias = nn.Parameter(torch.ones(n_visible) * 0.5, requires_grad=require_grad)
		# #applying a 0 bias to the hidden nodes
		self._hidden_bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=require_grad)

		#TODO cross check 
		# self.parameters=torch.nn.ParameterDict(
		# 	{
		# 		"weights": self._weights ,
		# 		"vis_bias": self._visible_bias,
		# 		"hid_bias": self._hidden_bias,
		# 	})

	#TODO these could be properties
	def get_visible_bias(self):
		return self._visible_bias

	def get_hidden_bias(self):
		return self._hidden_bias

	def get_weights(self):
		"""
		W = torch.zeros(self._n_visible, self._n_hidden)
		W_diff = torch.zeros(self._n_visible, self._n_hidden)
		W_diff[self._W_idxs[0], self._W_idxs[1]] = self._weights
		W = W + W_diff
		W = W + W.t()
		print("self._weights :", self._weights)
		"""
		return self._weights

	def run_training(self, in_data):
		""" Use sampler to train rbm. Each image from the current batch is used
		as input.
		"""
		loss=0
		for img in in_data:
			loss+=self.sampler.run_training(img)
		return loss	

	def get_logZ_value(self):
		#TODO include calculation of thois value
		# this hardcoded number is taken from Figure 10 Rolfe
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
		"""For the case in Discrete VAE paper, this is simply the log prob (verify!)"""
		return -self.log_prob(post_samples)

	def log_prob(self, samples):
		"""log(exp(-E(z))/Z) = -E(z) - log(Z)"""
		return -self.energy(samples)-self.get_logZ_value()


if __name__=="__main__":
	logger.info("Testing RBM Setup")

	n_batch_samples = 32
	VISIBLE_UNITS = 784  # 28 x 28 images
	HIDDEN_UNITS = 128
	N_GIBBS_SAMPLING_STEPS = 10
	n_epochs = 6
	N_EVENTS_TRAIN=-1
	N_EVENTS_TEST=-1
	do_train=False
	config_string="_".join(map(str,[N_EVENTS_TRAIN,n_epochs,N_GIBBS_SAMPLING_STEPS]))

	from data.mnist import get_mnist_datasets
	train_loader,test_loader=get_mnist_datasets(
			batch_size=n_batch_samples,
			num_evts_train=N_EVENTS_TRAIN,
			num_evts_test=N_EVENTS_TEST,
			binarise="threshold")
	
	rbm = RBM(n_visible=VISIBLE_UNITS, n_hidden=HIDDEN_UNITS, n_gibbs_sampling_steps=N_GIBBS_SAMPLING_STEPS)

	if do_train:
		for epoch in range(n_epochs):
			loss = 0    
			for batch_idx, (input_data, label) in enumerate(train_loader):
				loss_per_batch = rbm.train_sampler(input_data.view(-1,VISIBLE_UNITS))
				loss += loss_per_batch
			loss /= len(train_loader.dataset)
			logger.info('Epoch {0}. Loss={1:.2f}'.format(epoch,loss))
		import os
		logger.info("Saving Model")
		f=open("./output/rbm_test_200827_wdecay_{0}.pt".format(config_string),'wb')
		torch.save(rbm.sampler,f)
		f.close()
	else:
		# f=open("./output/rbm_test_200827_wdecay_{0}.pt".format(config_string),'rb')
		f=open("./output/divae_mnist/rbm_DiVAE_mnist_500_-1_100_1_0.001_4_100_RELU_default_201104.pt",'rb')
		rbm=torch.load(f)
# ))
		print(rbm)
		f.close()
	
	# # ########## EXTRACT FEATURES ##########
	logger.info("Sampling from RBM")
	for batch_idx, (input_data, label) in enumerate(test_loader):
		y=rbm.get_samples(input_data.view(-1,VISIBLE_UNITS))
		# energy=rbm.energy(input_data.view(-1,VISIBLE_UNITS))
		# print(energy)
		# cross_entropy=rbm.cross_entropy(input_data.view(-1,VISIBLE_UNITS))
		# print(cross_entropy)
		# use a random picture for sanity checks
		# samples=torch.rand(input_data.view(-1,VISIBLE_UNITS).size())
		# yrnd=rbm.get_samples(input_data.view(-1,VISIBLE_UNITS), random=True)
		# energy=rbm.energy(samples)
		# print(energy)
		break
	print(y.size())
	from utils.helpers import plot_MNIST_output

	# plot_MNIST_output(input_data,y, n_samples=5, output="./output/rbm_test_200827_wdecay_{0}.png".format(config_string))
	# plot_MNIST_output(input_data,yrnd, n_samples=5, output="./output/rbm_test_200827_rnd_{0}.png".format(config_string))

	# train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
	# train_labels = np.zeros(len(train_dataset))
	# test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
	# test_labels = np.zeros(len(test_dataset))
	
	# for i, (batch, labels) in enumerate(train_loader):
	#     batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
	#     train_features[i*n_batch_samples:i*n_batch_samples+len(batch)] = rbm.sample_from_hidden(batch).cpu().numpy()
	#     train_labels[i*n_batch_samples:i*n_batch_samples+len(batch)] = labels.numpy()
	
	# for i, (batch, labels) in enumerate(test_loader):
	#     batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
	#     test_features[i*n_batch_samples:i*n_batch_samples+len(batch)] = rbm.sample_from_hidden(batch).cpu().numpy()
	#     test_labels[i*n_batch_samples:i*n_batch_samples+len(batch)] = labels.numpy()
