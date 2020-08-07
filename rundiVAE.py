# -*- coding: utf-8 -*-
"""
Main runscript

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import sys
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
from helpers import plot_MNIST_output, gif_output

def run(tuner=None, config=None):
    tuner.load_data()

    model=None
    if config.type=="AE":
        model = AE(latent_dimensions=config.LATENT_DIMS)
    elif config.type=="VAE":
        model = VAE(latent_dimensions=config.LATENT_DIMS)
    elif config.type=="DiVAE":
        model = DiVAE(latent_dimensions=config.LATENT_DIMS)
    else:
        logger.error("Unknown Model Type")
        raise NotImplementedError

    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    model.print_model_info()

    tuner.register_model(model)
    tuner.register_optimiser(optimiser)

    gif_frames=[]
    logger.debug("Start Epoch Loop")
    for epoch in range(1, config.EPOCHS+1):
        train_loss = tuner.train(epoch)
        test_loss  = tuner.test()

        if config.create_gif:
            x_true, x_recon = tuner.evaluate()
            gif_frames.append(gif_output(x_true, x_recon,epoch=epoch, max_epochs=config.EPOCHS, train_loss=train_loss,test_loss=test_loss))
    
    if config.create_gif:
        configString="_".join(str(i) for i in [config.type,config.NUM_EVTS,config.BATCH_SIZE,config.EPOCHS,config.LEARNING_RATE,config.LATENT_DIMS])
        gif.save(gif_frames,"./output/200807_runs_{0}.gif".format(configString),duration=500)
    
    if config.create_plot:
        #possible to return many other parameters, x, x' just for plotting
        #evaluate should be more randomised!
        x_true, x_recon = tuner.evaluate()
        plot_MNIST_output(x_true, x_recon)

if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Willkommen")
    from argparse import ArgumentParser        
    argParser = ArgumentParser(add_help=False)
    argParser.add_argument( '-d', '--debug', help='Activate Debug Logging', action='store_true')
    argParser.add_argument( '-t', '--type', help='Switch between models: AE, VAE, DiVAE', default="AE")
    config=argParser.parse_args()
    #TODO make steerable
    config.NUM_EVTS = 1000
    config.BATCH_SIZE = 100 
    config.EPOCHS = 10
    config.LEARNING_RATE = 1e-3
    config.LATENT_DIMS = 2
    config.create_gif=False
    config.create_plot=False

    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("DEBUG MODE Activated")
    
    tuner=ModelTuner(config)
    
    run(tuner,config)

