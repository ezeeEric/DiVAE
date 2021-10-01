#!/usr/bin/python3
"""
Slurm executable. 
"""

#external libraries
import os
import time
import subprocess
import itertools

#self defined imports
from DiVAE import logging
logger = logging.getLogger(__name__)

COMMONPAR={
        "model.beta_smoothing_fct":[5,7,9],
        "model.output_smoothing_fct":[5,7,9],
        "engine.n_epochs":[75],
        }

TRAINPAR={"gamma":{
        "data.particle_type": ["gamma"],
        "data.scaler_path": ["/scratch/edrechsl/calogan/gamma_scaler.gz"],
        "data.scaler_amin": ["/scratch/edrechsl/calogan/gamma_amin.npy"],
        "engine.learning_rate":[0.00005],
        "engine.n_train_batch_size":[100,128],
        "engine.n_gibbs_sampling_steps":[60],
        **COMMONPAR,
        },
        "eplus":{
        "data.particle_type": ["eplus"],
        "data.scaler_path": ["/scratch/edrechsl/calogan/eplus_scaler.gz"],
        "data.scaler_amin": ["/scratch/edrechsl/calogan/eplus_amin.npy"],
        "engine.learning_rate":[0.0001],
        "engine.n_train_batch_size":[50,75,100],
        "engine.n_gibbs_sampling_steps":[50,60],
        **COMMONPAR,
        },
        "piplus":{
        "data.particle_type": ["piplus"],
        "data.scaler_path": ["/scratch/edrechsl/calogan/piplus_scaler.gz"],
        "data.scaler_amin": ["/scratch/edrechsl/calogan/piplus_amin.npy"],
        "engine.learning_rate":[0.0001,0.00005],
        "engine.n_train_batch_size":[50,75,100],
        "engine.n_gibbs_sampling_steps":[40,50],
        **COMMONPAR,
        },
        }

MODELPAR={
        "model": ["gumboltcaloV6_deep","gumboltcaloV6_deeper","gumboltcaloV6_deeper_big"],
        }
    
import itertools

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def dict_item_product(dicts):
    combinedString=[]
    for key, val in dicts.items():
        combinedString.append("{0}={1}".format(key,",".join([str(x) for x in val])))
    return " ".join(combinedString)

def submit():
    logger.info("Starting Submission")
    os.system("export WANDB_MODE=disabled")
    
    
    script="/home/edrechsl/codez/qVAE/DiVAE/scripts/run.py"
    
    for ptype, hpdict in TRAINPAR.items():
        HYPERDICT={**hpdict,**MODELPAR}
        cfgString=dict_item_product(HYPERDICT)
        #print("python {0} --config-name config_cedar --multirun {1} &".format(script,cfgString))
        os.system("python {0} --config-name config_cedar --multirun {1} &".format(script,cfgString))

    logger.info("Success")


if __name__=="__main__":
    submit()
