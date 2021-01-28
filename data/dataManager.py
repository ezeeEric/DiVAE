"""
Data Manager

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch

from DiVAE import logging
logger = logging.getLogger(__name__)
from DiVAE import config

from data.loadMNIST import loadMNIST
from data.loadCaloGAN import loadCalorimeterData

class DataManager(object):
    def __init__(self,train_loader=None,test_loader=None):
        self._train_loader=train_loader
        self._test_loader=test_loader

        self._input_dimensions=None
        self._train_dataset_mean=None
        return

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    def init_dataLoaders(self):
        logger.info("Loading Data")

        if not config.load_data_from_pkl:
            train_loader,test_loader=self.create_dataLoader()
        else:
            train_loader,test_loader,_,__=self.load_from_file()
        
        assert train_loader is not None, "Failed to set up train_loader"
        assert test_loader is not None, "Failed to set up train_loader"

        self._train_loader=train_loader
        self._test_loader=test_loader
        return

    def get_train_dataset_mean(self):
        return self._train_dataset_mean

    def get_input_dimensions(self):
        return self._input_dimensions

    def _set_input_dimensions(self):
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        input_sizes=self._train_loader.get_input_size()
        dataset[0][0].view(-1).size()[0]
        self._input_dimensions=input_sizes if isinstance(input_sizes,list) else [input_sizes]
        return

    def _set_train_dataset_mean(self):
        #returns mean of dataset as list
        #multiple input datasets - multiple means
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        
        input_dimensions=self.get_input_dimensions()
        imgPerLayer={}	
        for i in range(0,len(input_dimensions)):
            imgPerLayer[i]=[]	
        for i, (data, _) in enumerate(self._train_loader.dataset):
            #loop over all layers
            for l,d in enumerate(data):	
                imgPerLayer[l].append(d.view(-1,input_dimensions[l]))
        means=[]
        for l, imgList in imgPerLayer.items():
            means.append(torch.mean(torch.stack(imgList),dim=0))
        self._train_dataset_mean=means
        return 

    def pre_processing(self):
        if not config.load_data_from_pkl:
            self._set_input_dimensions()
            self._set_train_dataset_mean()
        else:
            _,__,input_dimensions,train_ds_mean=self.load_from_file()
            self._input_dimensions=input_dimensions
            self._train_dataset_mean=train_ds_mean
        return

    def create_dataLoader(self):
        if config.data_type.lower()=="mnist":
            train_dataset,test_dataset=loadMNIST(
                num_evts_train=config.n_train_samples,
                num_evts_test=config.n_test_samples, 
                binarise=config.binarise_dataset)

        elif config.data_type.lower()=="calo":
            inFiles={
                'gamma':    config.calo_input_gamma,
                'eplus':    config.calo_input_eplus,        
                'piplus':   config.calo_input_piplus         
            }
            train_dataset,test_dataset=loadCalorimeterData(
                inFiles=inFiles,
                ptype=config.particle_type,
                layers=config.calo_layers,
                num_evts_train=config.n_train_samples,
                num_evts_test=config.n_test_samples, 
                )
        #create the DataLoader for the training dataset
        train_loader=DataLoader(   
            train_dataset,
            batch_size=batch_size, 
            shuffle=True)
  
        #set batch size to full test dataset size - limitation only by hardware
        batch_size= len(test_dataset) if num_evts_test<0 else num_evts_test
        #create the DataLoader for the testing dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size, 
            shuffle=False)

        logger.debug("{0}: {2} events, {1} batches".format(train_loader,len(train_loader),len(train_loader.dataset)))
        logger.debug("{0}: {2} events, {1} batches".format(test_loader,len(test_loader),len(test_loader.dataset)))
        return train_loader,test_loader
    
    def load_from_file(self):
        #To speed up chain. Preprocessing involves loop over data for normalisation.
        #Load that data already prepped.
        import pickle
        with open(config.pre_processed_input_file, "rb") as dataFile:
            train_loader    =pickle.load(dataFile)
            test_loader     =pickle.load(dataFile)
            input_dimensions =pickle.load(dataFile)
            train_ds_mean   =pickle.load(dataFile)
        return train_loader, test_loader, input_dimensions, train_ds_mean