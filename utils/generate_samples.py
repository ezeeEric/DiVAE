import torch

import logging
logger = logging.getLogger(__name__)

def generate_samples_divae(model, outstring=""):
    n_samples=100
    output=model.generate_samples( n_samples=n_samples, n_gibbs_sampling_steps=100, sampling_mode="gibbs_flat")
    
    from utils.helpers import plot_generative_output
    plot_generative_output(output.detach(), n_samples=n_samples, output="./output/divae_mnist/rbm_samples/rbm_sampling_{0}.png".format(outstring))
    return

def generate_samples_vae(model, outstring=""):
    outputs=model.generate_samples(n_samples=50)
    outputs=outputs.detach()
    from utils.helpers import plot_generative_output
    plot_generative_output(outputs, n_samples=50, output="./output/vae_mnist/gen_samples/sampling_{0}.png".format(outstring))
    return

def generate_samples_cvae(model, outstring=""):
    nrs=[i for i in range(0,10)]
    outputs=model.generate_samples(n_samples_per_nr=5,nrs=nrs)
    outputs=outputs.detach()
    from utils.helpers import plot_generative_output
    plot_generative_output(outputs, n_samples=50, output="./output/cvae_mnist/sampling_{0}.png".format(outstring))
    return

def generate_samples_svae(model, outstring=""):
    outputs=model.generate_samples(n_samples=5)
    outputs=[ out.detach() for out in outputs]
    from utils.helpers import plot_calo_jet_generated
    plot_calo_jet_generated(outputs, n_samples=5, output="./output/svae_calo/generated_{0}.png".format(outstring))
    return

if __name__=="__main__":
    logger.info("Testing RBM Setup")