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

from modelTuner import ModelTuner
from diVAE import AE,VAE,DiVAE
from helpers import plot_MNIST_output, gif_output, plot_latent_space 

def run(tuner=None, config=None):
    tuner.load_data()

    enc_act_fct=torch.nn.ReLU() if config.ENC_ACT_FCT=="RELU" else None
    model=None
    if config.type=="AE":
        model = AE(latent_dimensions=config.LATENT_DIMS,encoder_activation_fct=enc_act_fct)
    elif config.type=="VAE":
        model = VAE(latent_dimensions=config.LATENT_DIMS,encoder_activation_fct=enc_act_fct)
    elif config.type=="DiVAE":
        model = DiVAE(latent_dimensions=config.LATENT_DIMS)
    else:
        logger.error("Unknown Model Type")
        raise NotImplementedError

    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    model.print_model_info()
    configString="_".join(str(i) for i in [config.type,config.NUM_EVTS_TRAIN,
                                        config.NUM_EVTS_TEST,config.BATCH_SIZE,
                                        config.EPOCHS,config.LEARNING_RATE,
                                        config.LATENT_DIMS,
                                        model.encoder.get_activation_fct()])

    tuner.register_model(model)
    tuner.register_optimiser(optimiser)

    if not config.load_model:
        gif_frames=[]
        logger.debug("Start Epoch Loop")
        for epoch in range(1, config.EPOCHS+1):   
            # lastEpoch=True if epoch==config.EPOCHS else False
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
        test_loss, x_true, x_recon, zetas, labels  = tuner.test()
        plot_latent_space(zetas, labels, output="{0}/200810_latSpace_{1}".format(config.output,configString),dimensions=0)
        plot_MNIST_output(x_true, x_recon, output="{0}/200810_reco_{1}.png".format(config.output,configString))

if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Willkommen")
    from argparse import ArgumentParser        
    argParser = ArgumentParser(add_help=False)
    argParser.add_argument( '-d', '--debug', help='Activate Debug Logging', action='store_true')
    argParser.add_argument( '-t', '--type', help='Switch between models: AE, VAE, DiVAE', default="AE")
    argParser.add_argument( '-o', '--output', help='Save Model here', default="./output/divae/")
    argParser.add_argument( '-i', '--infile', help='Load Model from this serialised file', default="./output/model_VAE_-1_500_100_10_0.001_2_ReLU.pt")

    config=argParser.parse_args()

    #TODO make steerable
    config.NUM_EVTS_TRAIN = 1000
    config.NUM_EVTS_TEST = 200
    config.BATCH_SIZE = 1000
    config.EPOCHS = 1
    config.LEARNING_RATE = 1e-3
    config.LATENT_DIMS = 2
    config.ENC_ACT_FCT="RELU"
    config.create_gif=False
    config.create_plots=False
    config.save_model=True
    config.load_model=not config.save_model
    config.sample_from_prior=False
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG MODE Activated")
    
    tuner=ModelTuner(config)
    if not os.path.exists(config.output):
        os.mkdir(config.output)
    tuner.outpath=config.output
    tuner.infile=config.infile
    run(tuner,config)

