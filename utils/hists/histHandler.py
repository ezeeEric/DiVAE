"""
Histogram handling framework
"""
import matplotlib.pyplot as plt
from coffea import hist
from io import BytesIO
from PIL import Image
import wandb

from utils.hists.totalEnergyHist import TotalEnergyHist

class HistHandler(object):
    
    def __init__(self, cfg):
        self._cfg = cfg
        self._hdict = {"totalEnergyHist":TotalEnergyHist()}
        
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
        
        value_dict = {cat_name:c_hist.values()[(cat_name,)] for cat_name in cat_names}
        
        fig = plt.figure()
        for cat_name in cat_names:
            plt.step(bins, value_dict[cat_name], label=cat_name)
            
        plt.legend(title=cat_ax.label)
        plt.xlabel(bin_ax.label)
        plt.ylabel(c_hist.label)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = wandb.Image(Image.open(buf))
        buf.close()
        
        return image