"""
ModelCreator - Interface between run scripts and models.

Provides initialisation of models.

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os
import torch

from DiVAE import logging
logger = logging.getLogger(__name__)

#import defined models
from models.autoencoders.autoencoder import AutoEncoder
from models.autoencoders.sparseAE import SparseAutoEncoder
from models.autoencoders.variationalAE import VariationalAutoEncoder
from models.autoencoders.hierarchicalVAE import HierarchicalVAE
from models.autoencoders.conditionalVAE import ConditionalVariationalAutoEncoder
from models.autoencoders.sequentialVAE import SequentialVariationalAutoEncoder
from models.autoencoders.discreteVAE import DiVAE
from models.autoencoders.dvaepp import DiVAEPP

_MODEL_DICT={
    "AE": AutoEncoder, 
    "sparseAE": SparseAutoEncoder,
    "VAE": VariationalAutoEncoder,
    "cVAE": ConditionalVariationalAutoEncoder,
    "sVAE": SequentialVariationalAutoEncoder,
    "HiVAE": HierarchicalVAE,
    "DiVAE": DiVAE,
    "DiVAEpp": DiVAEPP
}

class ModelCreator(object):

    def __init__(self, cfg=None):
        self._config=cfg

        self._model=None
        self._default_activation_fct=None
    
    def init_model(self,load_from_file=False, dataMgr=None):

        for key, model_class in _MODEL_DICT.items(): 
            if key.lower()==self._config.model.model_type.lower():
                logger.info("Initialising Model Type {0}".format(self._config.model.model_type))
                
                #TODO change init arguments. Ideally, the model does not carry
                #specific information about the dataset. 
                self.model=model_class(
                            flat_input_size=dataMgr.get_flat_input_size(),
                            train_ds_mean=dataMgr.get_train_dataset_mean(),
                            activation_fct=self._default_activation_fct,
                            cfg=self._config)

                return self.model
        logger.error("Unknown Model Type. Make sure your model is registered in modelCreator._MODEL_DICT.")
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

    def save_model(self,cfg_string='test'):
        logger.info("Saving Model")
        f=open(os.path.join(self._config.output_path,"model_{0}.pt".format(cfg_string)),'wb')
        torch.save(self._model.state_dict(),f)
        f.close()
        return

    def load_model(self):
        logger.info("Loading Model")
        #load_state_dict is a pytorch method (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        self._model.load_state_dict(torch.load(self._config.input_model))
        #set evaluation mode for model
        self._model.eval()
        return

if __name__=="__main__":
    logger.info("Willkommen!")
    mm=ModelCreator()
    logger.info("Success!")


