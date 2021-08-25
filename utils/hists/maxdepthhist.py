"""
Max Layer Depth histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class MaxDepthHist(object):
    def __init__(self, layer_dict):
        self._layer_dict = layer_dict
        n_layers = len(self._layer_dict.keys())
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("d", "Max Depth (GeV)", 0, n_layers-1, n_layers)))
        
    def update(self, in_data, recon_data, sample_data):
        return
        
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist