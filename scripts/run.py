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
    
    #instantiate the chosen model
    model=modelMaker.init_model()
    #create the NN infrastructure
    model.create_networks()
    model.print_model_info()

    #instantiate and register optimisation algorithm
    modelMaker.optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    #if we load a model from a file, we don't need to train
    if config.load_model:
        #TODO needs re-implementation
        # modelMaker.load_model(set_eval=True)
        #        if config.load_model:
        #   configString=config.infile.split("/")[-1].replace('.pt','')
 
        pass
    else:
        for epoch in range(1, config.n_epochs+1):   
            train_loss = modelMaker.fit(epoch=epoch, is_training=True)
            test_loss = modelMaker.fit(epoch=epoch, is_training=False)
    
    #TODO improve the save functionality
    if config.save_model:
        modelMaker.save_model(configString)
        if model.type=="DiVAE": 
            modelMaker.save_rbm(configString)

    exit()
    #TODO SAMPLE GENERATION - UNIFY, TEST
    if config.test_generate_samples:
        output_generated=modelMaker.generate_samples()

        #move this to PlotProvider 
        # if config.model_type=="DiVAE":  
        #     from utils.generate_samples import generate_samples_divae
        #     generate_samples_divae(modelMaker._model, configString)
        # #TODO split this up in plotting and generation routine and have one
        # #common function for all generative models. 
        # elif config.model_type=="VAE": 
        #     from utils.generate_samples import generate_samples_vae
        #     generate_samples_vae(modelMaker._model, configString)
        # elif config.model_type=="cVAE": 
        #     from utils.generate_samples import generate_samples_cvae
        #     generate_samples_cvae(modelMaker._model, configString)    
        # elif config.model_type=="sVAE": 
        #     from utils.generate_samples import generate_samples_svae
        #     generate_samples_svae(modelMaker._model, configString)

   #TODO PLOTTING
    from utils.helpers import plot_MNIST_output, gif_output, plot_latent_space, plot_calo_images, plot_calo_image_sequence

        gif_frames=[]
        for epoch in range(1, config.n_epochs+1):   
            test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
            if config.create_gif:
                #TODO improve
                if config.data_type=='calo':
                    gif_frames.append(plot_calo_images(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date),do_gif=True))
                else:
                    gif_frames.append(gif_output(input_data, output_data, epoch=epoch, max_epochs=config.n_epochs, train_loss=train_loss,test_loss=test_loss))
            
        if config.create_gif:
            gif.save(gif_frames,"{0}/runs_{1}.gif".format(config.output_path,configString),duration=200)
        if config.create_plots:
            if config.data_type=='calo':
                if config.model_type=="sVAE":
                    #TODO remove this
                    input_dimension=dataMgr.get_input_dimensions()
                    test_loss, input_data, output_data, zetas, labels   = modelMaker.test()
                    plot_calo_image_sequence(input_data, output_data, input_dimension, output="{0}/{2}_{1}.png".format(config.output_path,configString,date))
                else:
                    test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
                    plot_calo_images(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))
            else:
                test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
                if not config.model_type=="cVAE" and not config.model_type=="DiVAE":
                    plot_latent_space(zetas, labels, output="{0}/{2}_latSpace_{1}".format(config.output_path,configString,date),dimensions=0)
                plot_MNIST_output(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))

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

