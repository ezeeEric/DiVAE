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

	def load_model(self,set_eval=True):
		logger.info("Loading Model")
		#attention: model must be defined already
		self._model.load_state_dict(torch.load(self.infile))
		#training of model
		if set_eval:
			self._model.eval()
		return
	
	def load_data(self, binarise=True):
		logger.debug("Loading Data")

		self.train_loader,self.test_loader=loadMNIST(
			batch_size=self._config.BATCH_SIZE,
			num_evts_train=self._config.NUM_EVTS_TRAIN,
			num_evts_test=self._config.NUM_EVTS_TEST, 
			binarise=True)
		logger.debug("{0}: {2} events, {1} batches".format(self.train_loader,len(self.train_loader),len(self.train_loader.dataset)))
		logger.debug("{0}: {2} events, {1} batches".format(self.test_loader,len(self.test_loader),len(self.test_loader.dataset)))
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
		logger.debug("train")
		self._model.train()
		total_train_loss = 0
		for batch_idx, (x_true, label) in enumerate(self.train_loader):
			self._optimiser.zero_grad()
			x_true = torch.autograd.Variable(x_true)
			if self._model.type=='AE':
				x_recon, zeta = self._model(x_true)
				train_loss = self._model.loss(x_true,x_recon)
			elif self._model.type=='VAE':
				x_recon, mu, logvar, zeta = self._model(x_true)
				train_loss = self._model.loss(x_true, x_recon, mu, logvar)
			elif self._model.type=='DiVAE':
				x_recon, posterior_distribution, posterior_samples = self._model(x_true)
				#TODO continue here
				train_loss = self._model.loss(x_true, x_recon, posterior_distribution, posterior_samples)
			   
			train_loss.backward()
			total_train_loss += train_loss.item()
			self._optimiser.step()
			
			# Output logging
			if batch_idx % 10 == 0:
				logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx*len(x_true), len(self.train_loader.dataset),
					100.*batch_idx/len(self.train_loader), train_loss.data.item()/len(x_true)))
		logger.debug("finish train()")
		return total_train_loss/len(self.train_loader.dataset)

	def test(self):
		logger.debug("start test()")
		self._model.eval()
		test_loss = 0
		zeta_list=None
		label_list=None
		with torch.no_grad():
			for batch_idx, (x_true, label) in enumerate(self.test_loader):
				if self._model.type=='AE':
					x_recon, zeta = self._model(x_true)
					zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
					label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
					test_loss += self._model.loss(x_true,x_recon)
				elif self._model.type=='VAE':
					x_recon, mu, logvar, zeta = self._model(x_true)
					zeta_list=zeta.detach().numpy() if zeta_list is None else np.append(zeta_list,zeta.detach().numpy(),axis=0) 
					label_list=label.detach().numpy() if label_list is None else np.append(label_list,label.detach().numpy(),axis=0) 
					test_loss += self._model.loss(x_true, x_recon, mu, logvar)
				elif self._model.type=='DiVAE':
					x_recon, posterior_distribution, posterior_samples = self._model(x_true)
					#TODO continue here
					test_loss += self._model.loss(x_true, x_recon, posterior_distribution, posterior_samples)
				
		test_loss /= len(self.test_loader.dataset)
		logger.info("Test Loss: {0}".format(test_loss))
		logger.debug("finished test()")
		return test_loss, x_true, x_recon, zeta_list, label_list
