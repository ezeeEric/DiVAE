"""
Plotting functions for visualisation. 

All plotting functions must follow this structure:

def myPlotFunction(data_container: helpers.OutputContainer, cfg: hydra-config object):
    return None
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import importlib

from DiVAE import logging
logger = logging.getLogger(__name__)

def plot_shower_image_sequence(data_container=None, cfg=None):
    logger.info("plot_shower_image_sequence()")
    input_data=data_container.input_data
    output_data=data_container.outputs
    output_path=cfg.output_path
    n_samples=cfg.plotting.n_samples

    for i in range(n_samples):
        plt.figure(figsize=(10, 7))

        images=[]
        for j in range(0,len(input_data)):
            x=input_data[j]
            x_out=output_data[j]
            # plt.ylim([0,12])

            ax1 = plt.subplot(2, len(input_data), j + 1)
            ax1.set_box_aspect(1)
            if j==0:
                ax1.set_ylabel(r'$\phi$ Cell ID')

            im=plt.imshow(x[i],aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            images.append(im)
            reco_image=x_out[i].reshape(x[i].shape)
            
            #TODO this is an arbitrary hack to scale the values for a nice legend...
            minVal=reco_image.min(1,keepdim=True)[0]*15
            minVal=reco_image.min(1,keepdim=True)[0]

            reco_image[reco_image<minVal]=0
        
            ax2 = plt.subplot(2, len(input_data), j + 1 + len(input_data))
            ax2.set_box_aspect(1)
            if j==0:
                ax2.set_ylabel(r'$\phi$ Cell ID')
            ax2.set_xlabel(r'$\eta$ Cell ID')
            # else:
            #     ax2.get_yaxis().set_visible(False)

            im2=plt.imshow(reco_image,aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            # images.append(im2)

        fig = plt.gcf()
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = LogNorm(vmin=1e-5, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        cbar=fig.colorbar(images[0], ax=[fig.axes], orientation='vertical', fraction=.02)    
        cbar.ax.set_ylabel('Energy Fraction', rotation=270)

        for im in images:
            im.callbacksSM.connect('changed', update)
        fig.suptitle('Geant4 vs. sVAE Calorimeter shower')
        # plt.tight_layout()
        fig.savefig(output_path+"/test_{0}.png".format(i))

# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())

def plot_shower_metrics(data_container=None, cfg=None):
    logger.info("plot_shower_metrics()")

    logger.info("plot_shower_metrics()")
    
    moduleName="utils.plotting.showerMetrics"
    try:
        module=importlib.import_module(moduleName)
    except:
        logger.error("Could not import module {0} in PlotProvider.".format(moduleName))
        raise Exception
    
    metrics={}
    plot_metrics=cfg.plotting.plot_metrics
    for fct_key in plot_metrics:
        try:
            metrics[fct_key]=getattr(module, fct_key)
        except:
            raise Exception("Could not import function {0} from module {1} in PlotProvider.".format(fct_key,module))

    plotConfig={
        'total_energy': [(0,40,40),(r'$\gamma$    GEANT',r'$\gamma$    Reco'),'Total Energy [GeV]'],
    }
    
    orig_data=data_container.input_data
    #TODO currently the output data has dimensions [batch,x*y]. This ensures we
    #get [batch, x,y] and can use the same plotting functions. Can we change
    #this upstream?
    reco_data=[x.reshape(orig_data[idx].shape) for idx,x in enumerate(data_container.outputs)]
    for mName,mFct in metrics.items():
        plot_orig_data=mFct(orig_data).numpy()
        plot_reco_data=mFct(reco_data).numpy()
        bins = np.linspace(plotConfig[mName][0][0], plotConfig[mName][0][1], plotConfig[mName][0][2])
        _ = plt.hist(plot_orig_data, bins=bins, histtype='stepfilled', linewidth=2,
                        alpha=0.2,
                        label=plotConfig[mName][1][0])
        _ = plt.hist(plot_reco_data, bins=bins, histtype='step', linewidth=2,alpha=1,
                    label=plotConfig[mName][1][1])
        plt.legend(loc= 'upper right', ncol=1, fontsize=20)
        plt.xlabel(plotConfig[mName][2])
        plt.ylabel("Entries / Bin")
        plt.title(cfg.model.model_type)
        plt.savefig('{0}.png'.format(mName))


