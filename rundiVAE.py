# -*- coding: utf-8 -*-
"""
Main runscript

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os,sys
import pickle
from DiVAE import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(1)
import gif


from configaro import Configaro
from modelTuner import ModelTuner
from diVAE import AutoEncoder,VariationalAutoEncoder,HiVAE,DiVAE
from helpers import plot_MNIST_output, gif_output, plot_latent_space, plot_calo_images
from data.loadMNIST import loadMNIST
from data.loadCaloGAN import loadCalorimeterData

def load_data(config=None):
    logger.debug("Loading Data")

    train_loader,test_loader=None,None
    if config.dataType.lower()=="mnist":
        train_loader,test_loader=loadMNIST(
            batch_size=config.BATCH_SIZE,
            num_evts_train=config.NUM_EVTS_TRAIN,
            num_evts_test=config.NUM_EVTS_TEST, 
            binarise=config.binarise_dataset)

    elif config.dataType.lower()=="calo":
        inFiles={
            'gamma':    '/Users/drdre/inputz/CaloGAN_EMShowers/gamma.hdf5',
            'eplus':    '/Users/drdre/inputz/CaloGAN_EMShowers/eplus.hdf5',        
            'piplus':   '/Users/drdre/inputz/CaloGAN_EMShowers/piplus.hdf5'         
        }
        train_loader,test_loader=loadCalorimeterData(
            inFiles=inFiles,
            ptype=config.ptype,
            layer=config.caloLayer.lower(),
            batch_size=config.BATCH_SIZE,
            num_evts_train=config.NUM_EVTS_TRAIN,
            num_evts_test=config.NUM_EVTS_TEST, 
            )
    
    logger.debug("{0}: {2} events, {1} batches".format(train_loader,len(train_loader),len(train_loader.dataset)))
    logger.debug("{0}: {2} events, {1} batches".format(test_loader,len(test_loader),len(test_loader.dataset)))
    return train_loader,test_loader

def run(tuner=None, config=None):
    
    #load data, internally registers train and test dataloaders
    tuner.register_dataLoaders(*load_data(config=config))
    input_dimension=tuner.get_input_dimension()

    #set model properties
    model=None
    enc_act_fct=torch.nn.ReLU() if config.ENC_ACT_FCT=="RELU" else None    
    configString="_".join(str(i) for i in [config.type,config.dataType,config.NUM_EVTS_TRAIN,
                                        config.NUM_EVTS_TEST,config.BATCH_SIZE,
                                        config.EPOCHS,config.LEARNING_RATE,
                                        config.num_latent_hierarchy_levels,
                                        config.num_latent_units,
                                        config.ENC_ACT_FCT])

    if config.type=="AE":
        model = AutoEncoder(input_dimension=input_dimension,config=config,encoder_activation_fct=enc_act_fct)
    elif config.type=="VAE":
        model = VariationalAutoEncoder(input_dimension=input_dimension,config=config,encoder_activation_fct=enc_act_fct)
    elif config.type=="HiVAE":
        model = HiVAE(input_dimension=input_dimension,encoder_activation_fct=enc_act_fct,config=config)
    elif config.type=="DiVAE":
        model = DiVAE(config=config, n_hidden_units=config.N_HIDDEN_UNITS)
    else:
        logger.debug("ERROR Unknown Model Type")
        raise NotImplementedError

    model.print_model_info()

    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    tuner.register_model(model)
    tuner.register_optimiser(optimiser)

    if not config.load_model:
        gif_frames=[]
        logger.debug("Start Epoch Loop")
        for epoch in range(1, config.EPOCHS+1):   
            train_loss = tuner.train(epoch)       
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()

            if config.create_gif:
                gif_frames.append(gif_output(x_true, x_recon,epoch=epoch, max_epochs=config.EPOCHS, train_loss=train_loss,test_loss=test_loss))
            
            if model.type=="DiVAE" and config.sample_from_prior:
                random_samples=model.generate_samples()
                #TODO make a plot of the output

        if config.create_gif:
            gif.save(gif_frames,"./output/200807_runs_{0}.gif".format(configString),duration=500)
        
        if config.save_model:
            tuner.save_model(configString)
    else:
        tuner.load_model(set_eval=True)
        
    if config.create_plots:
        if config.dataType=='calo':
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()
            plot_calo_images(x_true, x_recon, output="{0}/200810_reco_{1}.png".format(config.output,configString))
        else:
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()
            plot_latent_space(zetas, labels, output="{0}/200810_latSpace_{1}".format(config.output,configString),dimensions=0)
            plot_MNIST_output(x_true, x_recon, output="{0}/201007_reco_{1}.png".format(config.output,configString))

if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Willkommen")

    config=Configaro()
    
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG MODE Activated")

    tuner=ModelTuner(config)
    
    if not os.path.exists(config.output):
        os.mkdir(config.output)
    tuner.outpath=config.output
    tuner.infile=config.infile
    
    run(tuner,config)

