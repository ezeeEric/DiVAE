"""
Fraction of energy per layer histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class FracTotalEnergyHist(object):
    def __init__(self, start_idx, end_idx, min_bin=1e-4, max_bin=1, n_bins=100):
        min_bin = 1e-4 if min_bin < 1e-4 else min_bin
        max_bin = 1 if max_bin > 1 else max_bin
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("f", "f",
                                              np.logspace(np.log10(min_bin), np.log10(max_bin), n_bins))))
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._scale = "log"
    
    def update(self, in_data, recon_data, sample_data):
        labels = ["input", "recon", "samples"]
        datasets = [in_data, recon_data, sample_data]
        layer_datasets = [dataset[:, self._start_idx:self._end_idx] for dataset in datasets]
        
        layer_datasets = [dataset.sum(axis=1) for dataset in layer_datasets]
        datasets = [dataset.sum(axis=1) for dataset in datasets]
        
        lfracs = [np.divide(layer_dataset, dataset) for layer_dataset, dataset in zip(layer_datasets, datasets)]
        
        for label, lfrac in zip(labels, lfracs):
            self._hist.fill(dataset=label, f=lfrac)
        
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale