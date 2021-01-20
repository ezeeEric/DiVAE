# -*- coding: utf-8 -*-
"""
Main runscript

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os,sys
import pickle
import datetime
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
from models.autoencoder import AutoEncoder
from models.sparseAE import SparseAutoEncoder
from models.variationalAE import VariationalAutoEncoder
from models.hierarchicalVAE import HierarchicalVAE
from models.conditionalVAE import ConditionalVariationalAutoEncoder
from models.sequentialVAE import SequentialVariationalAutoEncoder
from models.discreteVAE import DiVAE

from utils.helpers import plot_MNIST_output, gif_output, plot_latent_space, plot_calo_images, plot_calo_image_sequence
from data.loadMNIST import loadMNIST
from data.loadCaloGAN import loadCalorimeterData

def load_data(config=None):
    logger.debug("Loading Data")

    train_loader,test_loader=None,None
    if config.data_type.lower()=="mnist":
        train_loader,test_loader=loadMNIST(
            batch_size=config.n_batch_samples,
            num_evts_train=config.n_train_samples,
            num_evts_test=config.n_test_samples, 
            binarise=config.binarise_dataset)

    elif config.data_type.lower()=="calo":
        #TODO move to config
        inFiles={
            'gamma':    '/Users/drdre/inputz/CaloGAN_EMShowers/gamma.hdf5',
            'eplus':    '/Users/drdre/inputz/CaloGAN_EMShowers/eplus.hdf5',        
            'piplus':   '/Users/drdre/inputz/CaloGAN_EMShowers/piplus.hdf5'         
        }
        train_loader,test_loader=loadCalorimeterData(
            inFiles=inFiles,
            ptype=config.ptype,
            layers=config.calo_layerss,
            batch_size=config.n_batch_samples,
            num_evts_train=config.n_train_samples,
            num_evts_test=config.n_test_samples, 
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
        
        train_ds_mean=tuner.get_train_dataset_mean()

        # import pickle
        # dataFile=open("/Users/drdre/inputz/calo/preprocessed/all_la.pkl","wb")
        # pickle.dump(tuner.train_loader,dataFile)
        # pickle.dump(tuner.test_loader,dataFile)
        # pickle.dump(input_dimension,dataFile)
        # pickle.dump(train_ds_mean,dataFile)
        # dataFile.close()

    #set model properties
    model=None
    activation_fct=torch.nn.ReLU() if config.activation_fct.lower()=="relu" else None    
    
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
    
    date=datetime.datetime.now().strftime("%y%m%d")

    if config.data_type=='calo': 
        configString+="_nlayers_{0}_{1}".format(len(config.calo_layerss),config.ptype)

    #TODO wrap all these in a container class
    if config.model_type=="AE":
        if not config.sparse:
            model = AutoEncoder(input_dimension=input_dimension,config=config, activation_fct=activation_fct)
        else:
            model = SparseAutoEncoder(input_dimension=input_dimension,config=config, activation_fct=activation_fct)

    elif config.model_type=="VAE":
        model = VariationalAutoEncoder(input_dimension=input_dimension,config=config,activation_fct=activation_fct)
    
    elif config.model_type=="cVAE":
        model = ConditionalVariationalAutoEncoder(input_dimension=input_dimension,config=config,activation_fct=activation_fct)
    
    elif config.model_type=="sVAE":
        model = SequentialVariationalAutoEncoder(input_dimension=input_dimension,config=config,activation_fct=activation_fct)

    elif config.model_type=="HiVAE":
        model = HierarchicalVAE(input_dimension=input_dimension, activation_fct=activation_fct, config=config)

    elif config.model_type=="DiVAE":
        activation_fct=torch.nn.Tanh() 
        model = DiVAE(input_dimension=input_dimension, config=config, activation_fct=activation_fct)
    else:
        logger.debug("ERROR Unknown Model Type")
        raise NotImplementedError
    
    model.create_networks()
    model.set_dataset_mean(train_ds_mean)
    # model.set_input_dimension(input_dimension)

    #TODO avoid this if statement
    if config.model_type=="DiVAE": model.set_train_bias()

    model.print_model_info()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    tuner.register_model(model)
    tuner.register_optimiser(optimiser)
    
    if not config.load_model:
        gif_frames=[]
        logger.debug("Start Epoch Loop")
        for epoch in range(1, config.n_epochs+1):   
            train_loss = tuner.train(epoch)       
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()

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
            tuner.save_model(configString)
            if model.type=="DiVAE": 
                tuner.save_rbm(configString)

    else:
        tuner.load_model(set_eval=True)

    #TODO move this around
    if config.test_generate_samples:
        if config.load_model:
            configString=config.infile.split("/")[-1].replace('.pt','')
 
        if config.model_type=="DiVAE":  
            from utils.generate_samples import generate_samples_divae
            generate_samples_divae(tuner._model, configString)

        #TODO split this up in plotting and generation routine and have one
        #common function for all generative models. 
        elif config.model_type=="VAE": 
            from utils.generate_samples import generate_samples_vae
            generate_samples_vae(tuner._model, configString)

        elif config.model_type=="cVAE": 
            from utils.generate_samples import generate_samples_cvae
            generate_samples_cvae(tuner._model, configString)
        
        elif config.model_type=="sVAE": 
            from utils.generate_samples import generate_samples_svae
            generate_samples_svae(tuner._model, configString)

    if config.create_plots:
        if config.data_type=='calo':
            if config.model_type=="sVAE":
                test_loss, x_true, x_recon, zetas, labels   = tuner.test()
                plot_calo_image_sequence(x_true, x_recon, input_dimension, output="{0}/{2}_{1}.png".format(config.output_path,configString,date))
            else:
                test_loss, x_true, x_recon, zetas, labels  = tuner.test()
                plot_calo_images(x_true, x_recon, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))
        else:
            test_loss, x_true, x_recon, zetas, labels  = tuner.test()
            if not config.model_type=="cVAE" and not config.model_type=="DiVAE":
                plot_latent_space(zetas, labels, output="{0}/{2}_latSpace_{1}".format(config.output_path,configString,date),dimensions=0)
            plot_MNIST_output(x_true, x_recon, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))

if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Willkommen")

    config=Configaro()
    
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG MODE Activated")

    tuner=ModelTuner(config)
    
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    tuner.outpath=config.output_path
    tuner.infile=config.infile
    
    run(tuner,config)

