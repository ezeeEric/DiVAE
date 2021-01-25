# -*- coding: utf-8 -*-
"""
Discrete Variational Autoencoder Class

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os
import numpy as np

from DiVAE import logging
# from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config


class PlotProvider(object):
    def __init__(self):
        pass

    def plot(self):
           #TODO PLOTTING
    # from utils.helpers import plot_MNIST_output, gif_output, plot_latent_space, plot_calo_images, plot_calo_image_sequence

    #     gif_frames=[]
    #     for epoch in range(1, config.n_epochs+1):   
    #         test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
    #         if config.create_gif:
    #             #TODO improve
    #             if config.data_type=='calo':
    #                 gif_frames.append(plot_calo_images(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date),do_gif=True))
    #             else:
    #                 gif_frames.append(gif_output(input_data, output_data, epoch=epoch, max_epochs=config.n_epochs, train_loss=train_loss,test_loss=test_loss))
            
    #     if config.create_gif:
    #         gif.save(gif_frames,"{0}/runs_{1}.gif".format(config.output_path,configString),duration=200)
    #     if config.create_plots:
    #         if config.data_type=='calo':
    #             if config.model_type=="sVAE":
    #                 #TODO remove this
    #                 input_dimension=dataMgr.get_input_dimensions()
    #                 test_loss, input_data, output_data, zetas, labels   = modelMaker.test()
    #                 plot_calo_image_sequence(input_data, output_data, input_dimension, output="{0}/{2}_{1}.png".format(config.output_path,configString,date))
    #             else:
    #                 test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
    #                 plot_calo_images(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))
    #         else:
    #             test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
    #             if not config.model_type=="cVAE" and not config.model_type=="DiVAE":
    #                 plot_latent_space(zetas, labels, output="{0}/{2}_latSpace_{1}".format(config.output_path,configString,date),dimensions=0)
    #             plot_MNIST_output(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))
        
        # from utils.helpers import plot_generative_output
        # plot_generative_output(output.detach(), n_samples=n_samples, output="./output/divae_mnist/rbm_samples/rbm_sampling_{0}.png".format(outstring))

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
        pass

if __name__=="__main__":
    logger.debug("Testing PlotProvider Setup") 
    model=PlotProvider()
    logger.debug("Success")
