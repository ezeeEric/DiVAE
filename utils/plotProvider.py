# -*- coding: utf-8 -*-
"""
PlotProvider. Work in progress.

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import os
import numpy as np

from DiVAE import logging
# from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config

class PlotProvider(object):
    def __init__(self,config_string="default",date_tag="000000"):
        self.config_string=config_string
        self.date_tag=date_tag

        self._data_dimensions=None
        pass

    @property
    def data_dimensions(self):
        return self._data_dimensions

    @data_dimensions.setter
    def data_dimensions(self, dimensions):
        self._data_dimensions=dimensions

    def plot_generative_output(self):
        # from utils.helpers import plot_generative_output
        # plot_generative_output(output.detach(), n_samples=n_samples, output="./output/divae_mnist/rbm_samples/rbm_sampling_{0}.png".format(outstring))
        # def generate_samples_svae(model, outstring=""):
        #     outputs=model.generate_samples(n_samples=5)
        #     outputs=[ out.detach() for out in outputs]
        #     from utils.helpers import plot_calo_jet_generated
        #     plot_calo_jet_generated(outputs, n_samples=5, output="./output/svae_calo/generated_{0}.png".format(outstring))
        #     return
        pass

    def plot(self, input_container):
        logger.info("Plotting")
        #the container with our output objects to plot
        input_container.print()

        if config.data_type.lower()=="mnist":
            from utils.helpers import plot_MNIST_output, plot_latent_space
            
            #default plot method
            plot_MNIST_output(input_data=input_container.input_data, 
                output_data=input_container.output_data,
                n_samples=config.n_plot_samples, 
                out_file="{0}/{2}_reco_{1}.png".format(config.output_path,self.config_string,self.date_tag))
            
            #for visualisation purposes: plot latent spaces representation for certain models.
            #See method for details.
            if not config.model_type=="cVAE" and not config.model_type=="DiVAE":
                plot_latent_space(zeta=input_container.zetas,
                    label=input_container.labels, 
                    out_file="{0}/{2}_latSpace_{1}".format(config.output_path,self.config_string,self.date_tag),
                    dimensions=0)
        
        elif config.data_type.lower()=="calo":

            from utils.helpers import plot_calo_images, plot_calo_image_sequence
            
            if config.model_type=="sVAE":
                plot_calo_image_sequence(
                    input_data=input_container.input_data, 
                    output_data=input_container.output, 
                    input_dimension=self.data_dimensions, 
                    output="{0}/{2}_{1}.png".format(config.output_path,self.config_string,self.date_tag))
            else:
                test_loss, input_data, output_data, zetas, labels  = modelMaker.test()
                plot_calo_images(input_data, output_data, output="{0}/{2}_reco_{1}.png".format(config.output_path,configString,date))
        else:
            raise Exception("Data type {0} unknown to PlotProvider".format(config.data_type))
        
        #TODO 
        # gif_output, 
        # plot_latent_space
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
        pass

if __name__=="__main__":
    logger.debug("Testing PlotProvider Setup") 
    model=PlotProvider()
    logger.debug("Success")
