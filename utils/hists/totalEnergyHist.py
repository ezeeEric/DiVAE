"""
Class to handle the histograms aggregated during
training and plotted within wandb
"""
# Coffea histogramming library
from coffea import hist

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class TotalEnergyHist(object):
    
    def __init__(self, min_bin=0, max_bin=100, n_bins=100):
        self._hist = hist.Hist(label="Events",
                               axes=(hist.Cat("dataset", "dataset type"),
                                     hist.Bin("E", "Observed Energy (GeV)", n_bins, min_bin, max_bin)))
        
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