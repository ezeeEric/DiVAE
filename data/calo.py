"""Load up the MNIST data."""
import numpy as np
import torch
import h5py
from torch.utils.data import random_split, Dataset, Subset
from torchvision import transforms

from DiVAE import logging
logger = logging.getLogger(__name__)

#class wrapper containing Calorimeter images and returning energy as target
class CaloImage(object):
    def __init__(self, image, layer):        
        self._image=image
        self._layer=layer
        self._input_size=self._image[0].view(-1).size()[0]
        self._input_dimension=self._image[0].shape

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
    
    def get_input_dimension(self):
        return self._input_dimension

#contains images for all layers and the energy
class CaloImageContainer(Dataset):
    
    def __init__(self, particle_type=None, rawData=None, layer_subset=[]):
        self._particle_type=particle_type
        self._raw_data=rawData #hdf5 format

        self._dataset_size=len(self._raw_data["layer_0"])
        #dictionary of all calo images - keys are layer names
        self._images=None
        #true energy of the jets per event (same for all layers)
        self._true_energies=None
        #energy deposited outside the calorimeter layers - 1 value per layer.
        self._overflow_energies=None
        #the available events are split to train, test or val
        self._event_label=None

        #only use selected layers
        self._layer_subset=layer_subset

    def __len__(self):
        return len(self._true_energies)

    def __getattr__(self, item):
        """Only gets invoked if item doesn't exist in namespace.

        Args:
            item (): Requested output item
        """
        try:
            return self.__dict__[item]
        except:
            raise
 
    #pytorch dataloader needs this method
    def __getitem__(self,idx):
        #TODO this is divided by 100, because the CaloGAN dataset restricts
        #their jet energies to [0,100] GeV. This condition forces the energies
        #in range [0,1] which at the moment is needed for proper NN training
        #(i.e. sequentialVAE). Needs change, like norm to maximum in batch/full dataset.
        norm_true_energy=self._true_energies[idx]/100
        norm_overflow_energy=self._overflow_energies[idx]/100
        
        images=[]
        #if we request a subset of the calorimeter layers only
        if len(self._layer_subset)>0:
            for l in self._layer_subset:
                images.append(self._images[l]._get_image(idx))
        else:
            images=[img._get_image(idx) for l, img in self._images.items()]
        return images, (norm_true_energy, norm_overflow_energy)
    
    def get_input_size(self):
        sizes=[]
        for l,img in self._images.items():
            sizes.append(img.get_input_size())
        return sizes
    
    def get_input_dimension(self):
        dim=[]
        for l,img in self._images.items():
            dim.append(img.get_input_dimension())
        return dim
    
    def get_dataset_size(self):
        """Returns number of events in layer_0.
        """
        return self._dataset_size

    def process_data(self):
        calo_images={}
        for key, item in self._raw_data.items():
            #do not process energies here
            if key.lower() in ["energy","overflow"]: continue
            ds=self._raw_data[key][:]
            #convert df to CaloImage
            calo_images[key]=CaloImage(image=torch.Tensor(ds),layer=key)

        self._images=calo_images
        self._true_energies=self._raw_data["energy"][:]
        self._overflow_energies=self._raw_data["overflow"][:]

#Container for individual splits in train, test, val. This is necessary to work with random_split
#which creates Dataset.Subset instances, which lose all attributes of the
#initial class.
#TODO Is there a more elegant solution to this?
class CaloImageSubContainer(CaloImageContainer):

    def __init__(self, calo_subset=None, event_label=None):
        self.dataset=calo_subset.dataset
        self.indices=calo_subset.indices
        
        self._event_label=event_label     

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self,name):
        try:
            return self[name]
        except:
            if isinstance(self.dataset,CaloImageContainer):
                return self.dataset.__getattr__(name)
            elif isinstance(self.dataset.dataset,CaloImageContainer):
                return self.dataset.dataset.__getattr__(name)
            else:
                raise AttributeError()

def get_calo_datasets(inFiles={}, particle_type=["gamma"], layer_subset=[], frac_train_dataset=0.6, frac_test_dataset=0.2):
    
    #read in all input files for all jet types and layers
    dataStore={}
    for key,fpath in inFiles.items():     
        #for each particle_type, create a Container instance for our needs   
        dataStore[key]=CaloImageContainer(  particle_type=key,
                                            rawData=h5py.File(fpath,'r'),
                                            layer_subset=layer_subset)
        #convert image dataframes to tensors and get energies
        dataStore[key].process_data()

    assert len(particle_type)==1, "Currently only datasets for one particle type at a time\
         can be retrieved. Requested {0}".format(particle_type)
    ptype=particle_type[0]
    num_evts_total=dataStore[ptype].get_dataset_size()
    num_evts_train=int(frac_train_dataset*num_evts_total)

    #split in train and test
    train_subset,test_val_subset=random_split(dataStore[ptype],
                            [num_evts_train,num_evts_total-num_evts_train])
    #split in test and val
    num_evts_test_val=len(test_val_subset)
    num_evts_test=int(frac_test_dataset*num_evts_total)
    test_subset,val_subset=random_split(test_val_subset,
                            [num_evts_test,num_evts_test_val-num_evts_test])
    
    #create a container for each dataset to avoid working on pytorch Subsets.
    train_dataset=CaloImageSubContainer(train_subset, event_label="train")
    test_dataset=CaloImageSubContainer(test_subset, event_label="test")
    val_dataset=CaloImageSubContainer(val_subset, event_label="val")

    return train_dataset,test_dataset,val_dataset

if __name__=="__main__":
    inFiles={
        'gamma':    '/Users/drdre/inputz/CaloGAN_EMShowers/gamma.hdf5',
        'eplus':    '/Users/drdre/inputz/CaloGAN_EMShowers/eplus.hdf5',        
        'piplus':   '/Users/drdre/inputz/CaloGAN_EMShowers/piplus.hdf5'         
    }
    train_dataset,test_dataset,val_dataset = get_calo_datasets( 
                                                    particle_type=['gamma'], 
                                                    layer_subset=['layer_0','layer_1','layer_2'],
                                                    inFiles=inFiles, 
                                                    frac_train_dataset=0.6,
                                                    frac_test_dataset=0.2
                                                    )
    from torch.utils.data import DataLoader
    train_loader=DataLoader(   
    train_dataset,
    batch_size=10, 
    shuffle=True)                                       

    for batch_idx, (input_data, label) in enumerate(train_loader):
        print("Batch Idx: ", batch_idx)
        print("Number of images per event: ",len(input_data))
        print("Image shapes: ")
        print("Flat size:", train_dataset.get_input_size() )
        print("Shape:", train_dataset.get_input_dimension())
        # for x in input_data:
        #     print(x.get_input_size())
        #     print(x.get_input_dimension())
        print(label[0], label[1])
        exit()