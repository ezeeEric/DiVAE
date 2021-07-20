"""
Histogram handling framework
"""
import matplotlib.pyplot as plt
from coffea import hist
from io import BytesIO
from PIL import Image
import wandb

from utils.hists.totalEnergyHist import TotalEnergyHist
from utils.hists.diffEnergyHist import DiffEnergyHist
from utils.hists.layerEnergyHist import LayerEnergyHist
from utils.hists.fracTotalEnergyHist import FracTotalEnergyHist
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
            image_dict[hkey] = self.get_hist_image(self._hdict[hkey].get_hist())
        return image_dict
        
    def get_hist_image(self, c_hist):
        """
        Args:
            c_hist - coffea hist object
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
            
        ax.legend(title=cat_ax.label)
        ax.set_xlabel(bin_ax.label)
        ax.set_ylabel(c_hist.label)

        if bins[0]<0:
            ax.set_xscale('symlog')
        else:
            ax.set_xscale('log')
            
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = wandb.Image(Image.open(buf))
        buf.close()
        
        return image