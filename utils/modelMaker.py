"""
ModelMaker - Interface between run scripts and models.

Provides steering of training/testing/evaluation functionality.

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os
import torch
import numpy as np
#needs python3.7
from contextlib import nullcontext

from data.loadMNIST import loadMNIST

from DiVAE import logging
# from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config

#import defined models
from models.autoencoder import AutoEncoder
from models.sparseAE import SparseAutoEncoder
from models.variationalAE import VariationalAutoEncoder
from models.hierarchicalVAE import HierarchicalVAE
from models.conditionalVAE import ConditionalVariationalAutoEncoder
from models.sequentialVAE import SequentialVariationalAutoEncoder
from models.discreteVAE import DiVAE

model_dict={
    "AE": AutoEncoder, 
    "sparseAE": SparseAutoEncoder,
    "VAE": VariationalAutoEncoder,
    "cVAE": ConditionalVariationalAutoEncoder,
    "sVAE": SequentialVariationalAutoEncoder,
    "HiVAE": HierarchicalVAE,
    "DiVAE": DiVAE
}

class ModelMaker(object):
    def __init__(self):
        self._model=None
        self._optimiser=None

        self.data_mgr=None

        self._default_activation_fct=None
    
    def init_model(self,load_from_file=False):

        for key, model_class in model_dict.items(): 
            if key.lower()==config.model_type.lower():
                logger.info("Initialising Model Type {0}".format(config.model_type))
                #TODO change init arguments
                self.model=model_class(
                            input_dimension=self.data_mgr.get_input_dimensions(),
                            train_ds_mean=self.data_mgr.get_train_dataset_mean(),
                            activation_fct=self._default_activation_fct)
                
                # #return pre-trained model after loading from file
                # if load_from_file:
                #     return self.load_model()
                return self.model
        logger.error("Unknown Model Type. Make sure your model is registered in modelMaker.model_dict.")
        raise NotImplementedError

    @property
    def model(self):
        assert self._model is not None, "Model is not defined."
        return self._model

    @model.setter
    def model(self,model):
        self._model=model

    @property
    def default_activation_fct(self):
        return self._default_activation_fct

    @default_activation_fct.setter
    def default_activation_fct(self, act_fct):
        self._default_activation_fct=act_fct

    def save_model(self,config_string='test'):
        logger.info("Saving Model")
        f=open(os.path.join(config.output_path,"model_{0}.pt".format(config_string)),'wb')
        torch.save(self._model.state_dict(),f)
        f.close()
        return
    
    def save_rbm(self,config_string='test'):
        logger.info("Saving RBM")
        f=open(os.path.join(config.output_path,"rbm_{0}.pt".format(config_string)),'wb')
        torch.save(self._model.prior,f)
        f.close()
        return

    def register_dataManager(self,data_mgr):
        assert data_mgr is not None, "Empty Data Mgr"
        self.data_mgr=data_mgr
        return

    def load_model(self):
        logger.info("Loading Model")
        #load_state_dict is a pytorch method (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        self._model.load_state_dict(torch.load(config.input_model))
        #set evaluation mode for model
        self._model.eval()
        return

    @property
    def optimiser(self):
        return self._optimiser
    
    @optimiser.setter   
    def optimiser(self,optimiser):
        self._optimiser=optimiser

    def generate_samples(self):
        #options handled in model files
        #n_samples=100
        #n_gibbs_sampling_steps
        #sampling_mode
        #nrs
        #generate the samples. Each model has its own specific settings which
        #are read in from the config in the individual model files.
        output=self._model.generate_samples()
        #remove gradients and history, convert output to leaves so we can do
        #with it what we like (numpy operations for example).
        if isinstance(output,list):
            output=[out.detach() for out in output]
        else:
            output.detach()
        # outputs=model.generate_samples(n_samples=50)
        # nrs=[i for i in range(0,10)]
        # outputs=model.generate_samples(n_samples_per_nr=5,nrs=nrs)
        # outputs=model.generate_samples(n_samples=5)
        # outputs=[ out.detach() for out in outputs]
        return output

    def fit(self, epoch, is_training=True):
        logger.debug("Fitting model. Train mode: {0}".format(is_training))
        
        #set train/eval mode and context. torch.no_grad in eval mode improves
        #memory consumption. nullcontext is necessary to have a neat code
        #structure like below.
        if is_training:
            self._model.train()
            context=nullcontext()
            data_loader=self.data_mgr.train_loader
        else:
            self._model.eval()            
            context=torch.no_grad()
            data_loader=self.data_mgr.test_loader

        total_loss = 0
        with context:
            for batch_idx, (input_data, label) in enumerate(data_loader):
                #set gradients to zero before backprop. Needed in pytorch because
                #the default is to sum up gradients for successive backprop. steps.
                #that is useful for RNNs but not here.
                self._optimiser.zero_grad()
                #forward call
                #output is a namespace with members as added in the forward call
                #and subsequently used in loss()
                fwd_output=self._model(input_data)
                #loss call
                batch_loss = self._model.loss(input_data,fwd_output)
                
                if is_training:
                    batch_loss.backward()
                    self._optimiser.step()
                
                total_loss += batch_loss.item()

                # Output logging
                if is_training and batch_idx % 100 == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(
                                            epoch,
                                            batch_idx*len(input_data), 
                                            len(data_loader.dataset),
                                            100.*batch_idx/len(data_loader),
                                            batch_loss.data.item()/len(input_data)))
        
        outstring="Train" if is_training else "Test"
        total_loss /= len(data_loader.dataset)
        logger.info("Total Loss ({0}):\t {1:.4f}".format(outstring,total_loss))
        return total_loss
    
    def evaluate(self):
        #similar to test call of fit() method but returning values
        self._model.eval()
        #do plots on test dataset           
        data_loader=self.data_mgr.test_loader

        with torch.no_grad():
            for _, (input_data, label) in enumerate(data_loader):
                fwd_output=self._model(input_data)
                fwd_output.input_data=input_data
                fwd_output.labels = label
        return fwd_output

if __name__=="__main__":
    logger.info("Willkommen!")
    mm=ModelMaker()
    logger.info("Success!")


