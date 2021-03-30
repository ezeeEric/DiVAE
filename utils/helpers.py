"""
Unsorted helper functions

"""
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import gif
from DiVAE import logging
logger = logging.getLogger(__name__)

from types import SimpleNamespace

class OutputContainer(SimpleNamespace):
    """ #this is to facilitate a common interface in the ModelCreator fit()
        #method: instead of having different lengths of output arguments for
        #each model, we return one namespace. The entries of this namespace
        #are used in the model's loss function as parameter.
        This is based on types.SimpleNamespace but adds a fallback.
    """

    def __getattr__(self, item):
        """Only gets invoked if item doesn't exist in namespace.

        Args:
            item (): Requested output item
        """
        try:
            return self.__dict__[item]
        except KeyError: 
            logger.error("You requested a attribute {0} from the output object but it does not exist.".format(item))
            logger.error("Did you add the attribute in the forward() call of your method?")
            items = (f"{k}" for k, v in self.__dict__.items())
            logger.error("Available attributes: {0}".format("{}({})".format(type(self).__name__, ", ".join(items))))
            raise 

    def clear(self):
        """Clears the current namespace. Safety feature.
        """
        for key,_ in self.__dict__.items():
            self.__dict__[key]=None
        return self
    
    def print(self):
        """Clears the current namespace. Safety feature.
        """
        out=[str(key) for key,_ in self.__dict__.items()]
        logger.info("OutputContainer keys: {0}".format(out))
        