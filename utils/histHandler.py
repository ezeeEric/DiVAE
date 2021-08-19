"""
Histogram handling framework
"""
import matplotlib.pyplot as plt
from coffea import hist
from io import BytesIO
from PIL import Image
import wandb
import numpy as np

from utils.hists.totalEnergyHist import TotalEnergyHist
from utils.hists.diffEnergyHist import DiffEnergyHist
from utils.hists.layerEnergyHist import LayerEnergyHist
from utils.hists.fracTotalEnergyHist import FracTotalEnergyHist
from utils.hists.sparsityHist import SparsityHist
#from utils.hists.maxDepthHist import MaxDepthHist

_LAYER_SIZES={"layer_0" : [0, 288],
              "layer_1" : [288, 432],
              "layer_2" : [432, 504]}

class HistHandler(object):
    
    def __init__(self, cfg):
        self._cfg = cfg
        self._hdict = {"totalEnergyHist":TotalEnergyHist(),
                       "diffEnergyHist":DiffEnergyHist()}
        
        for layer in cfg.data.calo_layers:
            start_idx, end_idx = _LAYER_SIZES[layer]
            self._hdict[layer + "_EnergyHist"] = LayerEnergyHist(start_idx, end_idx)
            self._hdict[layer + "_fracEnergyHist"] = FracTotalEnergyHist(start_idx, end_idx)
            self._hdict[layer + "_sparsityHist"] = SparsityHist(start_idx, end_idx)
            
        layer_dict = {layer : _LAYER_SIZES[layer] for layer in cfg.data.calo_layers}
        #self._hdict["maxDepthHist"] = MaxDepthHist(layer_dict)
        
        
    def update(self, in_data, recon_data, sample_data):
        for hkey in self._hdict.keys():
            self._hdict[hkey].update(in_data, recon_data, sample_data)
            
    def clear(self):
        for hkey in self._hdict.keys():
            self._hdict[hkey].clear()
            
    def get_hist_images(self):
        """
        Returns:
            image_dict - Dict containing PIL images for each histogram
        """
        image_dict = {}
        for hkey in list(self._hdict.keys()):
            image_dict[hkey] = self.get_hist_image(self._hdict[hkey].get_hist(), self._hdict[hkey].get_scale())
        return image_dict
    
    def get_scatter_plots(self):
        """
        Returns:
            image_dict - Dict containing PIL images for each scatter plot
        """
        image_dict = {}
        for hkey in list(self._hdict.keys()):
            if "totalEnergyHist" in hkey or "_EnergyHist" in hkey:
                image_dict[hkey+"Scatter"] = self.get_scatter_plot(self._hdict[hkey].get_data_dict())
        return image_dict
        
    def get_hist_image(self, c_hist, scale='linear'):
        """
        Args:
            c_hist - coffea hist object
            scale - scale of the histogram (log, linear, ...)
        Returns:
            image - PIL image object
        """
        assert len(c_hist.axes()) == 2, "Histogram should only have two axes - Dataset type and Energy bins"
        ax_0, ax_1 = c_hist.axes()[0], c_hist.axes()[1]
        
        if isinstance(ax_0, hist.Cat) and isinstance(ax_1, hist.Bin):
            cat_ax = ax_0
            bin_ax = ax_1
        elif isinstance(ax_0, hist.Bin) and isinstance(ax_0, hist.Cat):
            bin_ax = ax_0
            cat_ax = ax_1
        else:
            raise ValueError("Expected categorical and bin axis")
        
        cat_names = [identifier.name for identifier in cat_ax.identifiers()]
        bins = [ax_bin.mid for ax_bin in bin_ax.identifiers()]
        
        value_dict = {cat_name:c_hist.values(overflow='all')[(cat_name,)] for cat_name in cat_names}
        bins = [bins[0]] + bins + [bins[len(bins)-1]]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for cat_name in cat_names:
            ax.step(bins, value_dict[cat_name], label=cat_name)
         
        ax.legend(title=cat_ax.label, prop={'size': 15})
        ax.set_xlabel(bin_ax.label, fontsize='15')
        ax.set_ylabel(c_hist.label, fontsize='15')
        ax.tick_params(axis='both', which='major', labelsize=15)

        if scale == 'log':
            ax.set_xscale('log')
            
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = wandb.Image(Image.open(buf))
        buf.close()
        plt.close()
        
        return image
    
    def get_scatter_plot(self, data_dict):
        """
        Args:
            data_dict - Dictionary with energy values
        
        Returns:
            image - PIL image object
        """
        assert ("input" in data_dict.keys() and "recon" in data_dict.keys())
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data_dict["input"], data_dict["recon"], alpha=0.5, marker='.')
        max_e = max(max(data_dict["input"]), max(data_dict["recon"]))
        ax.scatter(np.arange(1, max_e, 0.1), np.arange(1, max_e, 0.1), alpha=1., marker='.', c='r')
        ax.set_xlabel("Input Energy (GeV)", fontsize='15')
        ax.set_ylabel("Recon Energy (GeV)", fontsize='15')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlim(1, max_e)
        ax.set_ylim(1, max_e)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = wandb.Image(Image.open(buf))
        buf.close()
        plt.close()
        
        return image