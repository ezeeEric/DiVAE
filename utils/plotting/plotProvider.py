
"""
Plotting infrastructure to provide visualisations of input, reconstructed and
generated output.
"""

import os
import torch
import numpy as np
import importlib

from DiVAE import logging
logger = logging.getLogger(__name__)

class PlotProvider(object):
    def __init__(self, data_container=None, plotFunctions=['plot_MNIST_output'], cfg=None, **kwargs):
        
        self._config=cfg

        self._data_container=data_container

        self._data_dimensions=None
        
        #construct a list of plotting functions to be called from the
        #configuration. Imports a module which holds all plotting functions.
        self._plot_fcts=[]
        plot_fct_module=None
        try:
            plot_fct_module=importlib.import_module(self._config.plotting.plotModule)
        except:
            logger.error("Could not import module {0} in PlotProvider.".format(self._config.plotting.plotModule))
            raise Exception

        for it_fct in plotFunctions:
            try:
                self._plot_fcts.append(getattr(plot_fct_module, it_fct))
            except:
                logger.error("Could not import function {0} from module {1} in PlotProvider.".format(it_fct,self._config.plotting.plotModule))
                raise Exception
        pass

    @property
    def data_dimensions(self):
        return self._data_dimensions

    @data_dimensions.setter
    def data_dimensions(self, dimensions):
        self._data_dimensions=dimensions

    def plot(self, data_container=None):
        """This function calls all registered plotting functions.
        """
        for plt_fct in self._plot_fcts:
            plt_fct(self._data_container,self._config)

if __name__=="__main__":
    logger.debug("Testing PlotProvider Setup") 
    model=PlotProvider()
    logger.debug("Success")
