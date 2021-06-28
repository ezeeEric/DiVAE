"""
Wandb compatible plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from coffea import hist
from io import BytesIO
from PIL import Image
import wandb

_NORM_LIST = [LogNorm(vmax=10000, vmin=0.1), LogNorm(vmax=10000, vmin=0.1), LogNorm(vmax=10, vmin=0.1)]

def plot_calo_images(layer_images):
    image_list = []
    for idx in range(layer_images[0].shape[0]):
        image_list.append(plot_calo_image([layer_image[idx] for layer_image in layer_images]))
    return image_list
        
def plot_calo_image(image):
    fig, ax = plt.subplots(nrows=1, ncols=len(image), figsize=(12, 4))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=None)
    
    for layer, axe in enumerate(fig.axes):
        im = axe.imshow(image[layer], aspect='auto', origin='upper', norm=_NORM_LIST[layer])
        axe.set_title('Layer ' + str(layer), fontsize=10)
        axe.tick_params(labelsize=10)
        
        axe.set_yticks(np.arange(0, image[layer].shape[0], 1))
        axe.set_xlabel(r'$\phi$ Cell ID', fontsize=10)
        axe.set_ylabel(r'$\eta$ Cell ID', fontsize=10)
        
        if layer == 0:
            axe.set_xticks(np.arange(0, image[layer].shape[1], 10))
        else:
            axe.set_xticks(np.arange(0, image[layer].shape[1], 1))
            
        cbar = fig.colorbar(im, ax=axe)
        cbar.set_label('Energy, (MeV)', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=500)
    buf.seek(0)
    image = wandb.Image(Image.open(buf))
    buf.close()
    plt.close(fig)
        
    return image