"""Load up the MNIST data."""
import numpy as np
import torch
import h5py
import math 
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split

from torchvision import transforms
from DiVAE import logging
logger = logging.getLogger(__name__)

#class wrapper containing Calorimeter images and returning energy as target
class CaloImage(object):
    def __init__(self, image, layer):        
        self._image=image
        self._layer=layer
        self._input_size=self._image[0].view(-1).size()[0]

    def __len__(self):
        return len(self._image)
    
    #TODO this normalises the input data to [0,1]
    #needs to be changed for proper energy deposits.
    def normalise(self, img):
        minVal=img.view(-1,self._input_size).min(1,keepdim=True)[0]
        maxVal=img.view(-1,self._input_size).max(1,keepdim=True)[0]
        #if img all 0
        if abs(maxVal)>0:
            img-=minVal
            img/=maxVal
        return img

    def _get_image(self,idx):
        return self.normalise(self._image[idx])
    
    def get_input_size(self):
        return self._input_size

#contains images for all layers and the energy
class CaloImageContainer(Dataset):
    
    def __init__(self, images, energy):
        self._images=images
        self._layers=images.keys()
        self._isMultiLayer=True if len(self._layers)>1 else False
        self._energy=energy

    def __len__(self):
        return len(self._energy)

    def __getitem__(self,idx):
        #TODO this is divided by 100, because the CaloGAN dataset restricts
        #their jet energies to [0,100] GeV. This condition forces the energies
        #in range [0,1] which at the moment is needed for proper NN training
        #(i.e. sequentialVAE). Needs change, like norm to maximum in batch/full dataset.
        norm_energy=self._energy[idx]/100
        images=[img._get_image(idx) for l, img in self._images.items()]
        if not self._isMultiLayer:
            images=images[0]
            norm_energy=norm_energy[0]
        return images,norm_energy
    
    def get_input_size(self):
        sizes=[]
        for l,img in self._images.items():
            sizes.append(img.get_input_size())
        return sizes

#handles raw data processing
class CaloDataContainer(object):
    def __init__(self,ptype=None,rawData=None):
        self._ptype=ptype
        self._rawData=rawData #hdf5 format
        self._trainImageCont=None
        self._testImageCont=None

    def __getattr__(self,name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        else:
            return self._rawData[name]
    
    def processData(self,layers,item):
        trainImageDict,testImageDict={},{}
        trainEnergy_prev,testEnergy_prev=[],[]
        layers = [layers] if not isinstance(layers, list) else layers
        for l in layers:
            layerImages=self._rawData[l][:]
            energies=self._rawData[item][:]
            #This casts the numpy arrays into torch Tensors and combines this
            #into the CaloImage class defined above. 
            trI,teI,trainEnergy,testEnergy=train_test_split(layerImages,energies, test_size=0.2,shuffle=False)
            trainImages,testImages=map(lambda ds: CaloImage(image=torch.Tensor(ds),layer=l),[trI,teI])
            #sanity check whether energies between layers stay same
            if len(trainEnergy_prev)>0 and len(testEnergy_prev)>0:
                assert np.array_equal(trainEnergy_prev,trainEnergy) and np.array_equal(testEnergy_prev,testEnergy), "Energies between layers differ. This should not happen!"
            trainEnergy_prev=trainEnergy
            testEnergy_prev=testEnergy

            trainImageDict[l]=trainImages
            testImageDict[l] =testImages
        
        self._trainImageCont=CaloImageContainer(images=trainImageDict,energy=trainEnergy)
        self._testImageCont =CaloImageContainer(images=testImageDict,energy=testEnergy)

    def getDataset(self, train=True, num_evts=0):
        dataset=self._trainImageCont if train else self._testImageCont
        #restrict dataset size
        if num_evts>0:
            dataset=random_split(dataset,[num_evts, len(dataset)-num_evts])[0]
        return dataset

def loadCalorimeterData(inFiles={}, ptype='gamma', layers=['layer_1'], num_evts_train=0, num_evts_test=0):
    #read in all input files, sorted by jet type
    dataStore={}
    for key,fpath in inFiles.items():        
        dataStore[key]=CaloDataContainer(ptype=key,rawData=h5py.File(fpath,'r'))
    
    #for given particle type, retrieve data
    #TODO the whole chain is currently restricted to 1 ptype at a time.
    dataStore[ptype].processData(layers=layers,item="energy")
    train_dataset=dataStore[ptype].getDataset(train=True,num_evts=num_evts_train)
    test_dataset=dataStore[ptype].getDataset(train=False,num_evts=num_evts_test)

    return train_dataset,test_dataset

if __name__=="__main__":
    NUM_EVTS = 100
    n_batch_samples = 100
    n_epochs = 100
    learning_rate = 1e-3
    LATENT_DIMS = 32
    load_binarised_MNIST="threshold"

    inFiles={
        'gamma':    '/Users/drdre/inputz/CaloGAN_EMShowers/gamma.hdf5',
        'eplus':    '/Users/drdre/inputz/CaloGAN_EMShowers/eplus.hdf5',        
        'piplus':   '/Users/drdre/inputz/CaloGAN_EMShowers/piplus.hdf5'         
    }
    # train_loader,test_loader=loadCalorimeterData( ptype='gamma', layers=['layer_0'], inFiles=inFiles, batch_size=n_batch_samples,num_evts_train=NUM_EVTS,num_evts_test=NUM_EVTS)
    train_loader,test_loader=loadCalorimeterData( ptype='gamma', layers=['layer_0','layer_1','layer_2'], inFiles=inFiles, batch_size=n_batch_samples,num_evts_train=NUM_EVTS,num_evts_test=NUM_EVTS)
    print(train_loader.get_input_size())
    for batch_idx, (input_data, label) in enumerate(train_loader):
        print("Batch Idx: ", batch_idx)
        print("Number of images per event: ",len(input_data))
        print("Image shapes: ")
        for x in input_data:
            print(x.shape)
        print(label[0])
