"""
E_ratio histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class ERatioHist(object):
    def __init__(self, start_idx, end_idx, min_bin=0, max_bin=1, n_bins=50):
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("eratio", "eratio", n_bins, min_bin, max_bin)))
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._scale = "linear"
        
    def update(self, in_data, recon_data, sample_data):
        labels = ["input", "recon", "samples"]
        datasets = [in_data, recon_data, sample_data]
        
        layer_datasets = [dataset[:, self._start_idx:self._end_idx] for dataset in datasets]
        layer_eratios = []
        
        for layer_dataset in layer_datasets:
            layer_maxes = np.amax(layer_dataset, axis=1)
            layer_second_maxes = np.partition(layer_dataset, -2, axis=1)[:, -2]
            
            numer = layer_maxes - layer_second_maxes
            denom = layer_maxes + layer_second_maxes
            
            numer = numer[denom != 0]
            denom = denom[denom != 0]
            layer_eratios.append((numer/denom))
        
        for label, layer_eratio in zip(labels, layer_eratios):
            self._hist.fill(dataset=label, eratio=layer_eratio)
    
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale