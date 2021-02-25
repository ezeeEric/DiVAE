"""
Data Manager

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

import torch
from torch.utils.data import DataLoader

from DiVAE import logging
logger = logging.getLogger(__name__)

from data.mnist import get_mnist_datasets
from data.calo import get_calo_datasets

class DataManager(object):
    def __init__(self,train_loader=None,test_loader=None,val_loader=None, cfg=None):
        self._config=cfg
        
        self._train_loader=train_loader
        self._test_loader=test_loader
        self._val_loader=val_loader

        #this is a list of tensor.shape tuples (i.e.[(28,28)] for MNIST) 
        self._input_dimensions=None
        #list of flattened tensor.shape tuples (i.e. [784] for mnist)
        self._flat_input_sizes=None 

        self._train_dataset_means=None
        return

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    def init_dataLoaders(self):
        logger.info("Loading Data")

        if not self._config.load_data_from_pkl:
            train_loader,test_loader,val_loader=self.create_dataLoader()
        else:
            #TODO
            train_loader,test_loader,val_loader,_,__=self.load_from_file()
        
        assert bool(train_loader and test_loader and val_loader), "Failed to set up data_loaders"
        
        self._train_loader=train_loader
        self._test_loader=test_loader
        self._val_loader=val_loader
        return

    def get_train_dataset_mean(self):
        return self._train_dataset_mean

    def get_input_dimensions(self):
        return self._input_dimensions
    
    def get_flat_input_size(self):
        return self._flat_input_sizes

    def _set_input_dimensions(self):
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        self._input_dimensions=self._train_loader.dataset.get_input_dimensions()
    
    def _set_flattened_input_sizes(self):
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        self._flat_input_sizes=self._train_loader.dataset.get_flattened_input_sizes()

    def _set_train_dataset_mean(self):
        #TODO should this be the mean over the current batch only?
        #returns mean of dataset as list
        assert self._train_loader is not None, "Trying to retrieve datapoint from empty train loader"
        
        in_sizes=self.get_flat_input_size()
        imgPerLayer={}	
        
        #create an entry for each layer
        for i in range(0,len(in_sizes)):
            imgPerLayer[i]=[]	

        for i, (data, _) in enumerate(self._train_loader.dataset):
            #loop over all layers
            for l,d in enumerate(data):	
                imgPerLayer[l].append(d.view(-1,in_sizes[l]))
        means=[]
        for l, imgList in imgPerLayer.items():
            means.append(torch.mean(torch.stack(imgList),dim=0))

        self._train_dataset_mean=means

    def pre_processing(self):
        if not self._config.load_data_from_pkl:
            self._set_input_dimensions()
            self._set_flattened_input_sizes()
            self._set_train_dataset_mean()
        else:
            #TODO load from file
            raise NotImplementedError
            # _,__,input_dimensions,train_ds_mean=self.load_from_file()
            # self._input_dimensions=input_dimensions
            # self._train_dataset_mean=train_ds_mean

    def create_dataLoader(self):
        assert abs(self._config.data.frac_train_dataset-1)>=0, "Cfg option frac_train_dataset must be within (0,1]"
        assert abs(self._config.data.frac_test_dataset-0.99)>1.e-5, "Cfg option frac_test_dataset must be within (0,99]. 0.01 minimum for validation set"

        if self._config.data.data_type.lower()=="mnist":
            train_dataset,test_dataset,val_dataset=get_mnist_datasets(
                frac_train_dataset=self._config.data.frac_train_dataset,
                frac_test_dataset=self._config.data.frac_test_dataset, 
                binarise=self._config.data.binarise_dataset,
                input_path=self._config.data.mnist_input)

        elif self._config.data.data_type.lower()=="calo":
            inFiles={
                'gamma':    self._config.calo_input_gamma,
                'eplus':    self._config.calo_input_eplus,        
                'piplus':   self._config.calo_input_piplus         
            }
            train_dataset,test_dataset,val_dataset=get_calo_datasets(
                inFiles=inFiles,
                particle_type=[self._config.particle_type],
                layer_subset=self._config.data.calo_layers,
                frac_train_dataset=self._config.data.frac_train_dataset,
                frac_test_dataset=self._config.data.frac_test_dataset, 
                )
        #create the DataLoader for the training dataset
        train_loader=DataLoader(   
            train_dataset,
            batch_size=self._config.engine.n_batch_samples, 
            shuffle=True)

        #create the DataLoader for the testing/validation datasets
        #set batch size to full test/val dataset size - limitation only by hardware
        test_loader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset), 
            shuffle=False)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset), 
            shuffle=False)

        logger.info("{0}: {2} events, {1} batches".format(train_loader,len(train_loader),len(train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(test_loader,len(test_loader),len(test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(val_loader,len(val_loader),len(val_loader.dataset)))

        return train_loader,test_loader,val_loader
    
    def load_from_file(self):
        #To speed up chain. Preprocessing involves loop over data for normalisation.
        #Load that data already prepped.
        import pickle
        with open(self._config.pre_processed_input_file, "rb") as dataFile:
            train_loader    =pickle.load(dataFile)
            test_loader     =pickle.load(dataFile)
            input_dimensions =pickle.load(dataFile)
            train_ds_mean   =pickle.load(dataFile)
        return train_loader, test_loader, input_dimensions, train_ds_mean