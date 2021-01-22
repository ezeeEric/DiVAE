"""Load up the MNIST data."""
import torch

from torch.utils.data import DataLoader,Sampler,random_split,Dataset

#torchvision contains popular datasets, model architectures for computer vision
from torchvision import datasets, transforms
import random

class Binarise_Tensor_Bernoulli(object):
    def __init__(self):
        pass
    def __call__(self,indata):
        return torch.bernoulli(indata)

class Binarise_Tensor_Threshold(object):
    def __init__(self, threshold=0.5):
        self.threshold=threshold
        
    def __call__(self,indata):
        return torch.where((indata>self.threshold),torch.ones(indata.size()),torch.zeros(indata.size()))

class DataLoaderWithInputSize(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderWithInputSize,self).__init__(*args, **kwargs)
        self._input_size=self.dataset[0][0].view(-1).size()[0]
    
    def get_input_size(self):
        return self._input_size

def loadMNIST(batch_size, num_evts_train=0, num_evts_test=0, binarise=None):
    
    #this list of functions is applied to our input data in succession
    transform_functions=[transforms.ToTensor()]

    #convert mnist dataset from grayscale to binary
    #use Bernoulli distribution based sampling for binarisation

    if binarise.lower()=="bernoulli":
        transform_functions.append(Binarise_Tensor_Bernoulli())
    elif binarise.lower()=="threshold":
    #any pixel with threshold above X is activated
        transform_functions.append(Binarise_Tensor_Threshold(0.5))

    train_dataset_full=datasets.MNIST(
                        root='./data/', 
                        train=True, 
                        download=True, 
                        transform=transforms.Compose(transform_functions)
                    )
    test_dataset_full=datasets.MNIST(
                        root='./data/', 
                        train=False, 
                        transform=transforms.Compose(transform_functions)
                    )

    # allow to restrict dataset size
    train_dataset=train_dataset_full
    test_dataset=test_dataset_full
    if num_evts_train>0:
        train_dataset=random_split(
            train_dataset_full, 
            [num_evts_train, len(train_dataset_full)-num_evts_train])[0]

    if num_evts_test>0:
         test_dataset=random_split(
            test_dataset_full, 
            [num_evts_test, len(test_dataset_full)-num_evts_test])[0]


    train_loader=DataLoaderWithInputSize(   
        train_dataset,
        batch_size=batch_size, 
        shuffle=True)
  
    #set batch size to full test dataset size - limitation only by hardware
    batch_size= len(test_dataset) if num_evts_test<0 else num_evts_test
    test_loader = DataLoaderWithInputSize(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False)

    return train_loader,test_loader

if __name__=="__main__":
    NUM_EVTS = 100
    n_batch_samples = 20
    n_epochs = 100
    learning_rate = 1e-3
    LATENT_DIMS = 32
    load_binarised_MNIST="threshold"
    train_loader,test_loader=loadMNIST(batch_size=n_batch_samples,num_evts_train=NUM_EVTS,num_evts_test=NUM_EVTS,binarise=load_binarised_MNIST)
    for batch_idx, (input_data, label) in enumerate(test_loader):
        print(batch_idx,(len(input_data), label)) 
    # from helpers import plot_MNIST_output
    # plot_MNIST_output(input_data,input_data,output="./output/testbinarising.png")