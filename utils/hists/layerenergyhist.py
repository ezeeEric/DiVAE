"""
Per layer energy Histogram
"""

# Coffea histogramming library
from utils.hists.totalenergyhist import TotalEnergyHist

# DiVAE logging
from DiVAE import logging
logger = logging.getLogger(__name__)

class LayerEnergyHist(TotalEnergyHist):
    def __init__(self, start_idx, end_idx, min_bin=0, max_bin=100, n_bins=100):
        super(LayerEnergyHist, self).__init__(min_bin, max_bin, n_bins)
        self._start_idx = start_idx
        self._end_idx = end_idx
        
    def update(self, in_data, recon_data, sample_data):
        labels = ["input", "recon", "samples"]
        datasets = [in_data, recon_data, sample_data]
        datasets = [dataset[:, self._start_idx:self._end_idx] for dataset in datasets]
        datasets = [data.sum(axis=1) for data in datasets]
        
        for label, dataset in zip(labels, datasets):
            self._hist.fill(dataset=label, E=dataset)
            self._data_dict[label].extend(dataset)