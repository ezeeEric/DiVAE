"""
Total energy Histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class TotalEnergyHist(object):
    def __init__(self, min_bin=1, max_bin=500, n_bins=100):
        min_bin = 1e-2 if min_bin < 1e-2 else min_bin
        max_bin = 1 if max_bin < 1 else max_bin
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("E", "Observed Energy (GeV)",
                                              np.logspace(np.log10(min_bin), np.log10(max_bin), n_bins))))
        
    def update(self, in_data, recon_data, sample_data):
        labels = ["input", "recon", "samples"]
        datasets = [in_data, recon_data, sample_data]
        datasets = [data.sum(axis=1) for data in datasets]
        
        for label, dataset in zip(labels, datasets):
            self._hist.fill(dataset=label, E=dataset)
            
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist