"""
Contrastive Divergence sampling routine
"""

import torch
from torch import nn

from models.rbm.samplers.baseSampler import BaseSampler

class ContrastiveDivergence(BaseSampler):
	def __init__(self, **kwargs):
		super(ContrastiveDivergence, self).__init__(**kwargs)
		
		# #TODO these are the nn.Parameters of the RBM. It's not good design that
		# #these should be members of this class too. But they are required in the
		# #CD training.
		# self.rbm_weights = None
		# self.rbm_visible_bias = None
		# self.rbm_hidden_bias = None

		# NN training parameters adjusted during CD
		self.weights_update = None
		self.visible_bias_update = None
		self.hidden_bias_update = None

	def set_rbm_parameters(self):

		# NN training parameters adjusted during CD
		self.weights_update= torch.zeros(n_visible, n_hidden)
		self.visible_bias_update = torch.zeros(n_visible)
		self.hidden_bias_update = torch.zeros(n_hidden)

	def train(self, input_data):
		reco_data=self._contrastive_divergence(input_data):
		return loss(input_data,reco_data)

	def loss(self, input_data, reco_data):
		# Compute reconstruction error
		loss_fct = torch.nn.MSELoss(reduction='none')
		return loss_fct(input_data, reco_data).sum()

	def sample_from_hidden(self, probabilities_visible):
		output_hidden = torch.matmul(probabilities_visible, self._weights) + self._hidden_bias
		probabilities_hidden = torch.sigmoid(output_hidden)
		return probabilities_hidden

	def sample_from_visible(self, probabilities_hidden):
		output_visible = torch.matmul(probabilities_hidden, self._weights.t()) + self._visible_bias
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
		self.weights_update -= self._weights * self.weight_decay_factor  # L2 weight decay

		## simplified version of the same learning rule that uses the states of individual nodes
		self.visible_bias_update *= self.momentum_coefficient
		self.visible_bias_update += torch.sum(input_data - probabilities_visible_neg, dim=0)

		self.hidden_bias_update *= self.momentum_coefficient
		self.hidden_bias_update += torch.sum(probabilities_hidden_pos - probabilities_hidden_neg, dim=0)

		# batch_size = input_data.size(0)

		self._weights += self.weights_update * self.learning_rate #/ batch_size
		self._visible_bias += self.visible_bias_update * self.learning_rate #/ batch_size
		self._hidden_bias += self.hidden_bias_update * self.learning_rate #/ batch_size

		return probabilities_visible_neg

#Gibbs Sampler, manual implementation
class ContrastiveDivergence_old(nn.Module):
	def __init__(self, learning_rate, momentum_coefficient, n_gibbs_sampling_steps, weight_decay_factor, **kwargs):
		super(ContrastiveDivergence_old, self).__init__(**kwargs)

		# Hyperparameters for CDn training
		#TODO duplicated between RBM and CD classes - find better solution
		self.learning_rate = learning_rate
		self.momentum_coefficient = momentum_coefficient
		self.n_gibbs_sampling_steps = n_gibbs_sampling_steps
		self.weight_decay_factor = weight_decay_factor 
		
		self.n_visible=n_visible #from rbm
		self.n_hidden=n_hidden #from rbm

		# NN training parameters adjusted during CD
		self.weights_update= torch.zeros(n_visible, n_hidden)
		self.visible_bias_update = torch.zeros(n_visible)
		self.hidden_bias_update = torch.zeros(n_hidden)

		# random weights and biases for all layers
		# weights between visible and hidden nodes. 784x128 (that is 28x28 input
		#size, 128 arbitrary choice)
		# if requires_grad=False : we calculate the weight update ourselves, not
		# through backpropagation
		require_grad=True
		#arbitrarily scaled by 0.01 
		self._weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01, requires_grad=require_grad)
  		# #all biases initialised to 0.5
		self._visible_bias = nn.Parameter(torch.ones(n_visible) * 0.5, requires_grad=require_grad)
		# #applying a 0 bias to the hidden nodes
		self._hidden_bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=require_grad)
		
	def sample_from_hidden(self, probabilities_visible):
		output_hidden = torch.matmul(probabilities_visible, self._weights) + self._hidden_bias
		probabilities_hidden = torch.sigmoid(output_hidden)
		return probabilities_hidden

	def sample_from_visible(self, probabilities_hidden):
		output_visible = torch.matmul(probabilities_hidden, self._weights.t()) + self._visible_bias
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

	def contrastive_divergence_fixed_hid_vis(self, in_data):
		rbm_nodes_concat=torch.cat(in_data,dim=1).detach()
		n_split=rbm_nodes_concat.size()[1]//2
		positive_samples_left,positive_samples_right=torch.split(rbm_nodes_concat,split_size_or_sections=int(n_split),dim=1)

		output_visible_pos=positive_samples_left
		output_hidden_pos = positive_samples_right

		#TODO use sigmoid() and bernoulli here
		associations_pos = torch.matmul(output_visible_pos.t(), output_hidden_pos)

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
		self.weights_update -= self._weights * self.weight_decay_factor  # L2 weight decay

		## simplified version of the same learning rule that uses the states of individual nodes
		self.visible_bias_update *= self.momentum_coefficient
		self.visible_bias_update += torch.sum(output_visible_pos - probabilities_visible_neg, dim=0)

		self.hidden_bias_update *= self.momentum_coefficient
		self.hidden_bias_update += torch.sum(output_hidden_pos - probabilities_hidden_neg, dim=0)

		self._weights += self.weights_update * self.learning_rate #/ batch_size
		self._visible_bias += self.visible_bias_update * self.learning_rate #/ batch_size
		self._hidden_bias += self.hidden_bias_update * self.learning_rate #/ batch_size

		# Compute reconstruction error
		#loss = torch.sum((input_data - probabilities_visible_neg)**2)
		loss_fct = torch.nn.MSELoss(reduction='none')
		loss = loss_fct(output_visible_pos, probabilities_visible_neg).sum()
		return loss

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
		self.weights_update -= self._weights * self.weight_decay_factor  # L2 weight decay

		## simplified version of the same learning rule that uses the states of individual nodes
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


if __name__=="__main__":
	logger.info("Testing CD Setup")
	sampler=Contrastive_Divergence()
	logger.info("Success")