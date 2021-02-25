"""Load up the MNIST data."""
import torch
import random
import numpy as np
np.random.seed(69)

from torch.utils.data import random_split,Dataset,Subset

#torchvision contains popular datasets, model architectures for computer vision
from torchvision import datasets, transforms

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

#inherits from Subset class and adds two homebaked functions
class MNISTImageContainer(Subset):
    def __init__(self, subset=None):
        self.dataset=subset.dataset
        self.indices=subset.indices

    def get_flattened_input_sizes(self):
        #return flattened size of MNIST dataset example (784)
        return [self.dataset[0][0].view(-1).size()[0]]
    
    def get_input_dimensions(self):
        #return dimension of MNIST dataset example (28x28)
        return [self.dataset[0][0].squeeze().shape]

def get_mnist_datasets(frac_train_dataset=1, frac_test_dataset=0.9, binarise=None, input_path=""):

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
                        root=input_path, 
                        train=True, 
                        download=True, 
                        transform=transforms.Compose(transform_functions)
                    )
    test_dataset_full=datasets.MNIST(
                        root=input_path, 
                        train=False, 
                        transform=transforms.Compose(transform_functions)
                    )

    # allow to restrict dataset size
    train_dataset=train_dataset_full
    test_dataset=test_dataset_full
    validation_dataset=None

    if abs(frac_train_dataset)-1<1e-5:
        #requested a fraction of the dataset only. calculate how many events
        #that fraction is.
        num_evts_train=int(abs(frac_train_dataset)*len(train_dataset_full))
        train_dataset=random_split(
            train_dataset_full, 
            [num_evts_train, len(train_dataset_full)-num_evts_train])

        #we have split the dataset in two, but only want the first chunk
        train_dataset=train_dataset[0]

    if abs(frac_test_dataset)-1<1e-5:
        #requested a fraction of the dataset only. calculate how many events
        #that fraction is.
        num_evts_test=int(abs(frac_test_dataset)*len(test_dataset_full))

        test_dataset, validation_dataset =random_split(
            test_dataset_full, 
            [num_evts_test, len(test_dataset_full)-num_evts_test])

    train_dataset=MNISTImageContainer(subset=train_dataset)
    test_dataset=MNISTImageContainer(subset=test_dataset)
    validation_dataset=MNISTImageContainer(subset=validation_dataset)

    return train_dataset,test_dataset,validation_dataset

if __name__=="__main__":
    load_binarised_MNIST="threshold"
    train_dataset,test_dataset,validation_dataset=get_mnist_datasets(frac_train_dataset=1, frac_test_dataset=0.9, binarise=load_binarised_MNIST)
    print(len(train_dataset))
    print(len(test_dataset))
    print(len(validation_dataset))

    print(train_dataset.get_flattened_input_sizes())
    print(train_dataset.get_input_dimensions())
