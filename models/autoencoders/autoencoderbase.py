#!/usr/bin/env python

"""
Base class for Autoencoder frameworks.

Defines basic common methods and variables shared between models.
Each model overwrites as needed. 
This class inherits from torch.nn.Module, ensuring that network parameters
are registered properly. 
"""
import torch
import torch.nn as nn

#logging module with handmade settings.
from DiVAE import logging
logger = logging.getLogger(__name__)

from utils.helpers import OutputContainer

# Base Class for Autoencoder models
class AutoEncoderBase(nn.Module):
    def __init__(self, flat_input_size, train_ds_mean, activation_fct, cfg, **kwargs):
        """
        """

        super(AutoEncoderBase,self).__init__(**kwargs)
        #sanity checks
        if isinstance(flat_input_size,list):
            assert len(flat_input_size)>0, "Input dimension not defined, needed for model structure"
        else:
            assert flat_input_size>0, "Input dimension not defined, needed for model structure"
        # assert config is not None, "Config not defined"
        # assert config.model.n_latent_nodes is not None and config.model.n_latent_nodes>0, "Latent dimension must be >0"

        self._model_type=None
        """a short tag identifying the einput_dataact model, such as AE, VAE, diVAE...
        """

        # the main configuration namespace returned by configaro
        self._config=cfg

        # number of nodes in latent layer
        self._latent_dimensions=self._config.model.n_latent_nodes
        
        if len(flat_input_size)>1:
            logger.warning("Received multiple input dimension numbers. Assuming multiple inputs.")
            self._flat_input_size=flat_input_size
        else:
            self._flat_input_size=flat_input_size[0]

        self._activation_fct=activation_fct

        #TODO remove this dependency. It is only needed in the forward call of
        #the DiscreteVAE model. One solution could be to change the model_class
        #initialisation to digest **kwargs - flexible number of input args per model
        self._dataset_mean=train_ds_mean[0] if isinstance(train_ds_mean,list) else train_ds_mean

        self._output_container=OutputContainer()

    def type(self):
        """String identifier for current model.

        Returns:
            model_type: "AE", "VAE", etc.
        """
        return self._model_type

    def _create_encoder(self):
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def _create_decoder(self):
        raise NotImplementedError

    def _create_sampler(self):
        """
            Define the sampler to be used for sampling from the RBM.

        Returns:
            Instance of baseSampler.
        """
        raise NotImplementedError
    
    def generate_samples(self):
        raise NotImplementedError

    def __repr__(self):
        parameter_string="\n".join([str(par.shape) if isinstance(par,torch.Tensor) else str(par)  for par in self.__dict__.items()])
        return parameter_string
    
    def forward(self, input_data):
        """[summary]

        Args:
            input_data (): [aaa]

        Raises:
            NotImplementedError: [ccc]
        """
        raise NotImplementedError

    def print_model_info(self):
        for key,par in self.__dict__.items():
            if isinstance(par,torch.Tensor):
                logger.info("{0}: {1}".format(key, par.shape))
            else:
                logger.debug("{0}: {1}".format(key, par))
        
if __name__=="__main__":
    logger.info("Running autoencoderbase.py directly") 
    logger.info("Success")