"""
Total energy Histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

# Dataset labels
_LABELS = ["input", "recon", "samples"]

class TotalEnergyHist(object):
    def __init__(self, min_bin=1, max_bin=100, n_bins=100):
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("E", "Observed Energy (GeV)",
                                              n_bins, min_bin, max_bin)))
        self._scale = "linear"
        self._data_dict = {key:[] for key in _LABELS}
        
    def update(self, in_data, recon_data, sample_data):
        datasets = [in_data, recon_data, sample_data]
        datasets = [data.sum(axis=1) for data in datasets]
        
        for label, dataset in zip(_LABELS, datasets):
            self._hist.fill(dataset=label, E=dataset)
            self._data_dict[label].extend(dataset)
            
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale
    
    def get_data_dict(self):
        return self._data_dict