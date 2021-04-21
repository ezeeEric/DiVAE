#!/usr/bin/python3
"""
Run script for standalone AIS tests.
"""

#external libraries
import os

import torch
torch.manual_seed(1)
import numpy as np
import matplotlib.pyplot as plt
import hydra

from models.samplers.ais import AnnealedImportanceSampler
from models.samplers.exactRBMPartFctSolver import ExactPartitionSolver
from models.rbm.rbmBase import RBMBase

#self defined imports
from DiVAE import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs/sampler/", config_name="ais")
def main(cfg=None):

    #RBM
    rbm=None    
    if cfg.load_rbm_from_file:
        import pickle
        f=open(cfg.load_rbm_from_file ,'rb')
        eps=pickle.load(f)
        rbm=pickle.load(f)
        Z=pickle.load(f)
        logZ=pickle.load(f)
        f.close()   
    else:
        rbm=RBMBase(n_visible=cfg.n_visible, n_hidden=cfg.n_hidden)
        
        rbm.weights=torch.randn(cfg.n_visible, cfg.n_hidden)
        rbm.visible_bias=torch.rand(cfg.n_visible)
        rbm.hidden_bias=torch.rand(cfg.n_hidden)

    #EPS
    eps=None
    if cfg.run_exact_cross_check:
        eps=ExactPartitionSolver(rbm=rbm)
    
    #AIS
    ais=AnnealedImportanceSampler(target_rbm=rbm, n_betas=cfg.n_betas, n_ais_chains=cfg.n_ais_chains, n_gibbs_sampling_steps=10)

    runAIS(rbm=rbm, eps=eps, ais=ais, config=cfg)

def runAIS(rbm=None, eps=None, ais=None, config=None):
    logger.info("Running Partition function estimates.")
    logger.info("Exact Solver running...")
    
    eps_Z,eps_logZ=eps.calculatePartitionFct()
    
    logger.info("Annealed Importance Sampler running...")
    ais_logZ=ais.sample()

    logger.info("Results:\n\t EPS logZ: {0:.4f}\n\t AIS logZ: {1:.4f}".format(eps_logZ,ais_logZ.item()))


if __name__=="__main__":
    logger.info("Starting standalone Annealed Importance Sampling run.")

    main()

    logger.info("Auf Wiedersehen!")

