#!/usr/bin/python3
"""
Main executable. The run() method steers data loading, model creation, training
and evaluation by calling the respective interfaces.

Author: Abhishek <abhishek@myumanitoba.ca>
Author: Eric Drechsler <eric.drechsler@cern.ch>
"""

#external libraries
import os
import pickle
import datetime
import sys

import torch
torch.manual_seed(32)
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

# PyTorch imports
from torch import device, load, save
from torch.nn import DataParallel
from torch.cuda import is_available

# Add the path to the parent directory to augment search for module
sys.path.append(os.getcwd())
    
# Weights and Biases
import wandb

#self defined imports
from DiVAE import logging
logger = logging.getLogger(__name__)

from data.dataManager import DataManager
from utils.plotting.plotProvider import PlotProvider
from engine.engine import Engine
from models.modelCreator import ModelCreator

@hydra.main(config_path="../configs", config_name="config")
def main(cfg=None):

    #TODO hydra update: output path not needed anymore. Replace all instances
    #with current work directory instead. (Hydra sets that automatically)
    cfg.output_path=os.getcwd()

    wandb.init(entity="qvae", project="divae", config=cfg)  
    print(OmegaConf.to_yaml(cfg))
    
    #create model handling object
    modelCreator=ModelCreator(cfg=cfg)
    
    #run the ting
    run(modelCreator, config=cfg)

def run(modelCreator=None, config=None):

    #container for our Dataloaders
    dataMgr=DataManager(cfg=config)
    #initialise data loaders
    dataMgr.init_dataLoaders()
    #run pre processing: get/set input dimensions and mean of train dataset
    dataMgr.pre_processing()
    
    if config.model.activation_fct.lower()=="relu":
        modelCreator.default_activation_fct=torch.nn.ReLU()
    elif config.model.activation_fct.lower()=="tanh":
        modelCreator.default_activation_fct=torch.nn.Tanh()
    else:
        logger.warning("Setting identity as default activation fct")
        modelCreator.default_activation_fct=torch.nn.Identity() 

    #instantiate the chosen model
    #loads from file 
    model=modelCreator.init_model(load_from_file=config.load_model, dataMgr=dataMgr)
    #create the NN infrastructure
    model.create_networks()
    #Not printing much useful info at the moment to avoid clutter. TODO optimise
    model.print_model_info()
    
    # Load the model on the GPU if applicable
    dev = None
    if (config.device == 'gpu') and config.gpu_list:
        logger.info('Requesting GPUs. GPU list :' + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in list(config.gpu_list)]
        logger.info("Main GPU : " + devids[0])
        
        if is_available():
            print(devids[0])
            dev = device(devids[0])
            if len(devids) > 1:
                logger.info("Using DataParallel on {}".format(devids))
                model = DataParallel(model, device_ids=list(config.gpu_list))
            logger.info("CUDA available")
        else:
            dev = device('cpu')
            logger.info("CUDA unavailable")
    else:
        logger.info('Unable to use GPU. Switching to CPU.')
        dev = device('cpu')
        
    # Send the model to the selected device
    model.to(dev)

    engine=Engine(cfg=config)
    #add dataMgr instance to engine namespace
    engine.data_mgr=dataMgr
    #add device instance to engine namespace
    engine.device=dev

    # Log metrics with wandb
    wandb.watch(model)
    
    #instantiate and register optimisation algorithm
    engine.optimiser = torch.optim.Adam(model.parameters(), lr=config.engine.learning_rate)
    #add the model instance to the engine namespace
    engine.model = model

    #no need to train if we load from file.
    if config.load_model:
        #return pre-trained model after loading from file
        #Attention: the order here matters - the model must be initialised and
        #networks created. Internally, weights and biases of the NN are set to the
        #pretrained values but need to have been instantiated first.
        modelCreator.load_model()
        logger.info("Model loaded from file, skipping training.")
        pass
    else:
        for epoch in range(1, config.engine.n_epochs+1): 
            if config.train:
                train_loss = engine.fit(epoch=epoch, is_training=True)
            
            if config.test:
                test_loss = engine.fit(epoch=epoch, is_training=False)
    
    #save our trained model
    #also save the current configuration with the same tag for bookkeeping
    if config.save_model:
        #save our trained model
        date=datetime.datetime.now().strftime("%y%m%d")
        config_string="_".join(str(i) for i in [config.model.model_type,config.data.data_type,date,config.tag])
        modelCreator.save_model(config_string)

    if False:
        #call a forward method derivative - for output object.
        eval_output=engine.evaluate()
        
        #sample generation
        if config.generate_samples:
            output_generated=engine.generate_samples()
            eval_output.output_generated=output_generated

        #instantiate plotting infrastructure
        pp=PlotProvider(data_container=eval_output, plotFunctions=config.plotting.plotFunctions, config_string=config_string,date_tag=date, cfg=config)
        
        #TODO is there a neater integration than to add this as member?
        pp.data_dimensions=dataMgr.get_input_dimensions()
        
        #call all the registered plot functions (hydra config)
        pp.plot(eval_output)
    
    logger.info("run() finished successfully.")

if __name__=="__main__":
    logger.info("Starting main executable.")

    main()

    logger.info("Auf Wiedersehen!")

