#!/usr/bin/python3
"""
Slurm executable. 
"""

#external libraries
import os
import hydra
# Weights and Biases
import wandb

#self defined imports
from DiVAE import logging
logger = logging.getLogger(__name__)

from  scripts.run import run

@hydra.main(config_path="../configs", config_name="config")
def main(cfg=None):
    logger.info("Starting main()")
    #initialise wandb logging. Note that this function has many more options,
    #reference: https://docs.wandb.ai/ref/python/init
    #this function main() will be called multiple times in a hydra multirun.
    #therefore we need to set reinit=True and reset the output directory to the
    #current hydra dir
    #TODO it is possible to use grouped runs to view in wandb, this would be
    #useful for manual hyperpara optimisation
    current_run=wandb.init(entity="qvae", project="divae", dir=os.getcwd(), config=cfg, reinit=True)
    logger.info("Current hydra run: wandb run {0}".format(current_run.id))
    #run the ting
    run(config=cfg)

    #log and finalise current wandb run
    current_run.finish()
    logger.info("Success")


if __name__=="__main__":
    #this function is wrapped by hydra.main(). In a multirun, this function is
    #called as many times as the argument list requires. I.e. --multirun
    #config.myopt=1,2 calls the below
    main()
