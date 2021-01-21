# -*- coding: utf-8 -*-
"""
Main runscript

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

#external libraries
import os,sys
import pickle
import datetime
import gif

import torch
torch.manual_seed(1)
import numpy as np
import matplotlib.pyplot as plt

#self defined imports
from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config

from data.dataManager import DataManager

def run(modelMaker=None):

    #container for our Dataloaders
    dataMgr=DataManager()
    #initialise data loaders
    dataMgr.init_dataLoaders()
    #run pre processing: get/set input dimensions and mean of train dataset
    dataMgr.pre_processing()
    #add dataMgr instance to modelMaker namespace
    modelMaker.register_dataManager(dataMgr)

    input_dimension=dataMgr.get_input_dimensions()
    train_ds_mean=dataMgr.get_train_dataset_mean()

    #set parameters relevant for this run
    date=datetime.datetime.now().strftime("%y%m%d")

    configString="_".join(str(i) for i in [config.model_type,
                                        config.data_type,
                                        config.n_train_samples,
                                        config.n_test_samples,
                                        config.n_batch_samples,
                                        config.n_epochs,
                                        config.learning_rate,
                                        config.n_latent_hierarchy_lvls,
                                        config.n_latent_nodes,
                                        config.activation_fct,
                                        config.tag])
    
    if config.data_type=='calo': 
        configString+="_nlayers_{0}_{1}".format(len(config.calo_layers),config.particle_type)

    if config.activation_fct.lower()=="relu":
        modelMaker.default_activation_fct=torch.nn.ReLU() 
    elif config.activation_fct.lower()=="tanh":
        modelMaker.default_activation_fct=torch.nn.ReLU() 
    else:
        logger.warning("Setting identity as default activation fct")
        modelMaker.default_activation_fct=torch.nn.Identity() 
        
    model=modelMaker.init_model()
    exit()
    model.create_networks()
    model.print_model_info()

    exit()

    model.set_dataset_mean(train_ds_mean)
    # model.set_input_dimension(input_dimension)

    #TODO avoid this if statement
    if config.model_type=="DiVAE": model.set_train_bias()

    model.print_model_info()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    modelMaker.register_model(model)
    modelMaker.register_optimiser(optimiser)
    
    #TODO rewrite this as "as helpers"
    from utils.helpers import plot_MNIST_output, gif_output, plot_latent_space, plot_calo_images, plot_calo_image_sequence

    if not config.load_model:
        gif_frames=[]
        logger.debug("Start Epoch Loop")
        for epoch in range(1, config.n_epochs+1):   
            train_loss = modelMaker.train(epoch)       
            test_loss, x_true, x_recon, zetas, labels  = modelMaker.test()

            if config.create_gif:
                #TODO improve
                if config.data_type=='calo':
                    gif_frames.append(plot_calo_images(x_true, x_recon, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date),do_gif=True))
                else:
                    gif_frames.append(gif_output(x_true, x_recon, epoch=epoch, max_epochs=config.n_epochs, train_loss=train_loss,test_loss=test_loss))
            
            if model.type=="DiVAE" and config.sample_from_prior:
                random_samples=model.generate_samples()
                #TODO make a plot of the output

        if config.create_gif:
            gif.save(gif_frames,"{0}/runs_{1}.gif".format(config.output_path,configString),duration=200)
        
        if config.save_model:
            modelMaker.save_model(configString)
            if model.type=="DiVAE": 
                modelMaker.save_rbm(configString)

    else:
        modelMaker.load_model(set_eval=True)

    #TODO move this around
    if config.test_generate_samples:
        if config.load_model:
            configString=config.infile.split("/")[-1].replace('.pt','')
 
        if config.model_type=="DiVAE":  
            from utils.generate_samples import generate_samples_divae
            generate_samples_divae(modelMaker._model, configString)

        #TODO split this up in plotting and generation routine and have one
        #common function for all generative models. 
        elif config.model_type=="VAE": 
            from utils.generate_samples import generate_samples_vae
            generate_samples_vae(modelMaker._model, configString)

        elif config.model_type=="cVAE": 
            from utils.generate_samples import generate_samples_cvae
            generate_samples_cvae(modelMaker._model, configString)
        
        elif config.model_type=="sVAE": 
            from utils.generate_samples import generate_samples_svae
            generate_samples_svae(modelMaker._model, configString)

    if config.create_plots:
        if config.data_type=='calo':
            if config.model_type=="sVAE":
                test_loss, x_true, x_recon, zetas, labels   = modelMaker.test()
                plot_calo_image_sequence(x_true, x_recon, input_dimension, output="{0}/{2}_{1}.png".format(config.output_path,configString,date))
            else:
                test_loss, x_true, x_recon, zetas, labels  = modelMaker.test()
                plot_calo_images(x_true, x_recon, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))
        else:
            test_loss, x_true, x_recon, zetas, labels  = modelMaker.test()
            if not config.model_type=="cVAE" and not config.model_type=="DiVAE":
                plot_latent_space(zetas, labels, output="{0}/{2}_latSpace_{1}".format(config.output_path,configString,date),dimensions=0)
            plot_MNIST_output(x_true, x_recon, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))

if __name__=="__main__":
    logger.info("Starting main executable.")

    #check if output path exists, create if necessary
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    
    #create model handling object
    from utils.modelMaker import ModelMaker
    modelMaker=ModelMaker()

    #run the ting
    run(modelMaker)

    logger.info("Auf Wiedersehen!")

