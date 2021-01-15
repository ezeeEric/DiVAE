# -*- coding: utf-8 -*-
"""
Runscript specifically for sVAE with Calo data. Development, might be merged
with main script later.

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

from utils.configaro import Configaro
from utils.modelTuner import ModelTuner
from models.sequentialVAE import SequentialVariationalAutoEncoder
from utils.helpers import gif_output, plot_calo_images
from data.loadCaloGAN import loadCalorimeterData

def load_data(config=None):
    logger.debug("Loading Data")

    train_loader,test_loader=None,None
    if config.dataType.lower()=="calo":
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
    
    if config.load_data_from_pkl:
        #To speed up chain. Postprocessing involves loop over data for normalisation.
        #Load that data already prepped.

        import pickle
        dataFile=open("/Users/drdre/inputz/MNIST/preprocessed/full.pkl","rb")
        train_loader=pickle.load(dataFile)
        test_loader =pickle.load(dataFile)
        input_dimension=pickle.load(dataFile)
        train_ds_mean=pickle.load(dataFile)
        dataFile.close()
        tuner.register_dataLoaders(train_loader, test_loader)
    else:
        #load data, internally registers train and test dataloaders
        tuner.register_dataLoaders(*load_data(config=config))
        input_dimension=tuner.get_input_dimension()
        train_ds_mean=tuner.get_train_dataset_mean(input_dimension)
        # import pickle
        # dataFile=open("/Users/drdre/inputz/calo/preprocessed/full_layer0.pkl","wb")
        # pickle.dump(tuner.train_loader,dataFile)
        # pickle.dump(tuner.test_loader,dataFile)
        # pickle.dump(input_dimension,dataFile)
        # pickle.dump(train_ds_mean,dataFile)
        # dataFile.close()

    #set model properties
    model=None
    activation_fct=torch.nn.ReLU() if config.activation_fct.lower()=="relu" else None    
    configString="_".join(str(i) for i in [config.type,config.dataType,config.NUM_EVTS_TRAIN,
                                        config.NUM_EVTS_TEST,config.BATCH_SIZE,
                                        config.EPOCHS,config.LEARNING_RATE,
                                        config.num_latent_hierarchy_levels,
                                        config.num_latent_units,
                                        config.activation_fct,
                                        config.tag])
    if config.dataType=='calo': 
        configString+="_{0}_{1}".format(config.caloLayer,config.ptype)
        
    #TODO wrap all these in a container class
    if config.type=="AE":
        model = AutoEncoder(input_dimension=input_dimension,config=config, activation_fct=activation_fct)
        
    elif config.type=="VAE":
        model = VariationalAutoEncoder(input_dimension=input_dimension,config=config,activation_fct=activation_fct)
    
    elif config.type=="cVAE":
        model = ConditionalVariationalAutoEncoder(input_dimension=input_dimension,config=config,activation_fct=activation_fct)
    
    elif config.type=="sVAE":
        model = SequentialVariationalAutoEncoder(input_dimension=input_dimension,config=config,activation_fct=activation_fct)

    elif config.type=="HiVAE":
        model = HierarchicalVAE(input_dimension=input_dimension, activation_fct=activation_fct, config=config)

    elif config.type=="DiVAE":
        activation_fct=torch.nn.Tanh() 
        model = DiVAE(input_dimension=input_dimension, config=config, activation_fct=activation_fct)
    else:
        logger.debug("ERROR Unknown Model Type")
        raise NotImplementedError
    
    model.create_networks()
    model.set_dataset_mean(train_ds_mean,input_dimension)
    #TODO avoid this if statement
    if config.type=="DiVAE": model.set_train_bias()

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
                #TODO improve
                if config.dataType=='calo':
                    gif_frames.append(plot_calo_images(x_true, x_recon, output="{0}/200810_reco_{1}.png".format(config.output,configString),do_gif=True))
                else:
                    gif_frames.append(gif_output(x_true, x_recon, epoch=epoch, max_epochs=config.EPOCHS, train_loss=train_loss,test_loss=test_loss))
            
            if model.type=="DiVAE" and config.sample_from_prior:
                random_samples=model.generate_samples()
                #TODO make a plot of the output

        if config.create_gif:
            gif.save(gif_frames,"{0}/runs _{1}.gif".format(config.output,configString),duration=200)
        
        if config.save_model:
            tuner.save_model(configString)
            if model.type=="DiVAE": 
                tuner.save_rbm(configString)

    else:
        tuner.load_model(set_eval=True)

        #TODO move this around
    if config.test_generate_samples:
        if config.load_model:
            configString=config.infile.split("/")[-1].replace('.pt','')
 
        if config.type=="DiVAE":  
            from utils.generate_samples import generate_samples,generate_iterative_samples
            # generate_samples(tuner._model)
            generate_iterative_samples(tuner._model, configString)

        #TODO split this up in plotting and generation routine and have one
        #common function for all generative models. 
        elif config.type=="VAE": 
            from utils.generate_samples import generate_samples_vae
            generate_samples_vae(tuner._model, configString)

        elif config.type=="cVAE": 
            from utils.generate_samples import generate_samples_cvae
            generate_samples_cvae(tuner._model, configString)
        
    if config.create_plots:
        if config.dataType=='calo':
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()
            plot_calo_images(x_true, x_recon, output="{0}/200810_reco_{1}.png".format(config.output,configString))
        else:
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()
            # plot_latent_space(zetas, labels, output="{0}/200810_latSpace_{1}".format(config.output,configString),dimensions=0)
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

