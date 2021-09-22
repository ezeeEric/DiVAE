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

#from scripts.run import run
#learning_rate: 0.001
#n_train_batch_size: 128
#n_epochs: 50
#n_gibbs_sampling_steps: 40
#kl_annealing_ratio: 0.3
#particle_type: gamma
#frac_train_dataset: 0.7
#frac_val_dataset: 0.1
#scaled: True
#scaler_path: /scratch/edrechsl/calogan/gamma_scaler.gz
#scaler_amin: /scratch/edrechsl/calogan/gamma_amin.npy
#scaler_path: 0
#scaler_amin: 0


PARTICLEPAR={
        #"data.particle_type":["eplus","piplus"],["gamma"]#
        "data.particle_type":["gamma"]#
        }


COMMONPAR={
        "model.beta_smoothing_fct":[5,7,9],
        "model.output_smoothing_fct":[1,5,10],
        }

TRAINPAR={"gamma":{
        "engine.learning_rate":[0.0001],
        "engine.n_train_batch_size":[42,64,86],
        "engine.n_epochs":[75,100],
        "engine.n_gibbs_sampling_steps":[40],
        **COMMONPAR,
        },
        "eplus":{
        "engine.learning_rate":[0.01,0.005,0.0001,0.00005],
        "engine.n_train_batch_size":[64,128,192],
        "engine.n_epochs":[75],
        "engine.n_gibbs_sampling_steps":[20,40,60],
        **COMMONPAR,
        },
        "piplus":{
        "engine.learning_rate":[0.01,0.005,0.0001,0.00005],
        "engine.n_train_batch_size":[64,128,192],
        "engine.n_epochs":[75],
        "engine.n_gibbs_sampling_steps":[20,40,60],
        **COMMONPAR,
        },
        }

MODELPAR={
        "model": ["gumboltcaloV","gumboltcaloV6","gumboltcaloV6_deep"],
        }
    
#    "activation_function":"relu",
    #    "model.beta_smoothing_fct":[0.1,5,10],
    #    "model.beta_smoothing_fct":[7],
    #    "decoder_hidden_nodes":[[50,100,100],[200,300,400],[400,600,800]],
    #    "encoder_hidden_nodes":[[100,100,50,],[400,300,200],[800,600,400]],
    #    "model.n_encoder_layer_nodes":[128,400],
    #    "model.n_encoder_layers":[2,4,8],
    #    "model.n_latent_hierarchy_lvls": [2,4,8],
    #    "model.n_latent_nodes": [128,256],
        #"engine.n_gibbs_sampling_steps":[40],
        #"engine.kl_annealing_ratio":[0.1,0.3,0.5],
        #"engine.kl_annealing_ratio":[0.3],

HYPERPARDICT={**TRAINPAR,**MODELPAR}
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
    #os.system("sh copyData.sh")
    #os.system("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK")
    os.system("export WANDB_MODE=disabled")
    
    cfgString=dict_item_product(HYPERPARDICT)
    
    script="/home/edrechsl/codez/qVAE/DiVAE/scripts/run.py"
    
    #run the ting
    #print("python {0} --multirun {1}".format(script,cfgString))
    #os.system("python {0} --config-name config_cedar --multirun {1}".format(script,cfgString))
    
    for key, val in PARTICLEPAR.items():
        for v in val:
            particleTypeCfg="{0}={1}".format(key,v)
#            print(' '.join(["python",script,"--config-name","config_cedar","--multirun",cfgString,"{0}={1}".format(key,v)]))
#            subprocess.Popen(["python",script,"--config-name","config_cedar","--multirun",cfgString],shell=False, stdout=subprocess.PIPE)
            os.system("python {0} --config-name config_cedar --multirun {1} {2} &".format(script,cfgString,particleTypeCfg))
#            time.sleep(5)

    logger.info("Success")


if __name__=="__main__":
    submit()
