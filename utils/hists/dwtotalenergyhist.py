"""
Depth-weighted total energy histogram
"""

# Coffea histogramming library
from coffea import hist
import numpy as np

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

# Dataset labels
_LABELS = ["input", "recon", "samples"]

class DWTotalEnergyHist(object):
    def __init__(self, layer_dict, n_bins=100):
        self._layer_dict = layer_dict
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("dw", "Depth-weighted total energy",
                                              np.logspace(np.log10(1), np.log10(10**3), n_bins))))
        self._scale = "log"
        
    def update(self, in_data, recon_data, sample_data):
        labels = ["input", "recon", "samples"]
        datasets = [in_data, recon_data, sample_data]
        
        layer_energies = {}
        for layer, layer_idxs in self._layer_dict.items():
            layer_energies[layer] = [dataset[:, layer_idxs[0]:layer_idxs[1]].sum(axis=1) for dataset in datasets]
            
        layer_energies_keys = list(layer_energies.keys())
        
        curr_datasets = layer_energies[layer_energies_keys[0]]
        curr_datasets = [curr_dataset * 0. for curr_dataset in curr_datasets]
        
        for idx in range(1, len(layer_energies_keys)):
            layer = layer_energies_keys[idx]
            curr_datasets = [curr_dataset+(layer_dataset*(idx)) for layer_dataset,curr_dataset in zip(layer_energies[layer],curr_datasets)]
            
        for dataset in curr_datasets:
            dataset = dataset.reshape(-1)
            
        for label, curr_dataset in zip(labels, curr_datasets):
            self._hist.fill(dataset=label, dw=curr_dataset)
        
    def clear(self):
        self._hist.clear()
        
    def get_hist(self):
        return self._hist
    
    def get_scale(self):
        return self._scale