"""
Input Energy - Recon Energy Histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class DiffEnergyHist(object):
    def __init__(self, edge_bin=50, n_bins=100):
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("diff_E", "Diff Energy (Input-Recon) (GeV)", n_bins, -edge_bin, edge_bin)))
        self._scale = "linear"
        
    def update(self, in_data, recon_data, sample_data):
        datasets = [in_data, recon_data]
        datasets = [data.sum(axis=1) for data in datasets]
        self._hist.fill(dataset="calo", diff_E=(datasets[0]-datasets[1]))
            
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale