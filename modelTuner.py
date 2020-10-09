import os
import torch
import numpy as np

from data.loadMNIST import loadMNIST

from DiVAE import logging
logger = logging.getLogger(__name__)

class ModelTuner(object):
	def __init__(self, config=None):
		self._config=config
		self._model=None
		self._optimiser=None

		self.train_loader=None
		self.test_loader=None

		self.outpath=""
		self.infile=""

	def save_model(self,config_string='test'):
		logger.info("Saving Model")
		f=open(os.path.join(self.outpath,"model_{0}.pt".format(config_string)),'wb')
		torch.save(self._model.state_dict(),f)
		f.close()
		return

	def register_dataLoaders(self,train_loader,test_loader):
		self.train_loader=train_loader
		self.test_loader=test_loader
		return
	
	def get_input_dimension(self):
		assert self.train_loader is not None, "Trying to retrieve datapoint from empty train loader"
    	#this gets the first member of the dataset, flattens it and return its
    	#size. Needed for model construction.
		return self.train_loader.dataset[0][0].view(-1).size()[0]


	def load_model(self,set_eval=True):
		logger.info("Loading Model")
		#attention: model must be defined already
		self._model.load_state_dict(torch.load(self.infile))
		#training of model
		if set_eval:
			self._model.eval()
		return
	
	def register_model(self,model):
		logger.debug("Register Model")
		self._model=model
		return

	def register_optimiser(self,optimiser):
		logger.debug("Register Model")
		self._optimiser=optimiser
		return

	def train(self, epoch):
		logger.info("Training Model")
		#set pytorch train mode
		self._model.train()

		total_train_loss = 0
		for batch_idx, (x_true, label) in enumerate(self.train_loader):
			#set gradients to zero before backprop. Needed in pytorch
			self._optimiser.zero_grad()

			#each of the architectures implement slightly different forward
			#calls and loss functions
			if self._config.type=='AE':
				x_recon, zeta = self._model(x_true)
				train_loss = self._model.loss(x_true,x_recon)

			elif self._config.type=='VAE':
				x_recon, mu, logvar, zeta = self._model(x_true)
				train_loss = self._model.loss(x_true, x_recon, mu, logvar)	

			elif self._config.type=='HiVAE':
				x_recon, mu_list, logvar_list, zeta_list = self._model(x_true)
				train_loss = self._model.loss(x_true, x_recon, mu_list, logvar_list)	

			elif self._config.type=='DiVAE':
				x_recon, output_activations, output_distribution,\
						 posterior_distribution, posterior_samples = self._model(x_true)
				train_loss = self._model.loss(x_true, x_recon, output_activations, output_distribution, posterior_distribution, posterior_samples)
			else:
				logger.debug("ERROR Unknown Model Type")
				raise NotImplementedError

			train_loss.backward()
			total_train_loss += train_loss.item()
			self._optimiser.step()
			
			# Output logging
			if batch_idx % 100 == 0:
				logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx*len(x_true), len(self.train_loader.dataset),
					100.*batch_idx/len(self.train_loader), train_loss.data.item()/len(x_true)))
		
		total_train_loss /= len(self.train_loader.dataset)
		logger.info("Train Loss: {0}".format(total_train_loss))
		return total_train_loss
	
	def test(self):
		logger.info("Testing Model")
		self._model.eval()

		test_loss = 0
		zeta_list=None
		label_list=None

		with torch.no_grad():
			for batch_idx, (x_true, label) in enumerate(self.test_loader):
				if self._config.type=='AE':
					x_recon, zeta = self._model(x_true)
					test_loss += self._model.loss(x_true,x_recon)
					
					#for plotting
					zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
					label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
				
				elif self._config.type=='VAE':
					x_recon, mu, logvar, zeta = self._model(x_true)
					test_loss += self._model.loss(x_true, x_recon, mu, logvar)
					
					#for plotting
					zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
					label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
				
				elif self._config.type=='HiVAE':
					x_recon, mu_list, logvar_list, zeta_hierarchy_list = self._model(x_true)
					test_loss += self._model.loss(x_true, x_recon, mu_list, logvar_list)
					for zeta in zeta_hierarchy_list:
						zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
					label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
				
				elif self._config.type=='DiVAE':
					x_recon, output_activations, output_distribution,\
						 posterior_distribution, posterior_samples = self._model(x_true)
					#TODO continue here
					test_loss += self._model.loss(x_true, x_recon, output_activations, output_distribution, posterior_distribution, posterior_samples)
				
		test_loss /= len(self.test_loader.dataset)
		logger.info("Test Loss: {0}".format(test_loss))
		return test_loss, x_true, x_recon, zeta_list, label_list
