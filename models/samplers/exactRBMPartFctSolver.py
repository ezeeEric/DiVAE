"""
Exact solutions for Partition Function estimates. 
"""

import numpy as np
import torch
from torch import nn
from copy import deepcopy

from models.rbm.rbm import RBM
from models.samplers.baseSampler import BaseSampler

from DiVAE import logging
logger = logging.getLogger(__name__)

class ExactPartitionSolver(object):
    def __init__(self, rbm=None, **kwargs):
        assert rbm is not None, "ExactPartitionSolver needs an RBMBase instance"

        self._rbm=rbm
        self._hid_bin_vec=[]
        self._vis_bin_vec=[]

        self.createInputStates(rbm.n_visible,rbm.n_hidden)

    def createInputStates(self, n_visible, n_hidden):
        
        for b in range(2**n_visible):
            bx=bin(b)[2:].zfill(n_visible)
            intList=[int(bx[i:i+1]) for i in range(n_visible)] 
            self._vis_bin_vec.append(intList)

        for b in range(2**n_hidden):
            bx=bin(b)[2:].zfill(n_hidden)
            intList=[int(bx[i:i+1]) for i in range(n_hidden)] 
            self._hid_bin_vec.append(intList)  

    def calculatePartitionFct(self):
        act_list=[]
        for it_hid in self._hid_bin_vec:
            hid=torch.IntTensor(it_hid)
            for it_vis in self._vis_bin_vec:
                vis=torch.IntTensor(it_vis)

                visxw=torch.matmul(self._rbm.weights,vis.float())
                hwv=torch.matmul(hid.float().t(),visxw)
                act=hwv+torch.matmul(self._rbm.visible_bias,vis.float())+torch.matmul(self._rbm.hidden_bias,hid.float())
                act_list.append(act)
        Z=torch.sum(torch.exp(-torch.Tensor(act_list)))
        logZ=torch.log(torch.sum(torch.exp(-torch.Tensor(act_list))))
        return Z,logZ

    def calculatePartitionFct(self):
        #pydeep adapted...
        act_list=[]
        hid=torch.FloatTensor(self._hid_bin_vec)
        act=torch.matmul(hid,self._rbm.weights.t())+self._rbm.visible_bias
        bias=torch.matmul(hid,self._rbm.hidden_bias.t()).reshape(act.shape[0],1)
        logph=bias+torch.sum(torch.log(torch.exp(act)+1),dim=1).reshape(act.shape[0],1)
        Z=logph
        logZ=torch.log(torch.sum(torch.exp(logph)))
        return Z,logZ

    def loadFromFile(self):
        import pickle
        f=open("/Users/drdre/codez/qVAE/DiVAE/output/210324_eps/rbm_1010.pkl",'rb')
        # Exact Partition Fct for RBM(10,10)
        # Z=4886438400.0
        # logZ=22.309728622436523
        eps=pickle.load(f)
        Z=pickle.load(f)
        logZ=pickle.load(f)
        f.close()

if __name__=="__main__": 
    logger.info("Loading Model")
    # rbm=RBM(n_visible=200,n_hidden=200)
    # input_rbm="/Users/drdre/codez/qVAE/DiVAE/outputs/2021-03-17/10-54-18/rbm.pt"
    # rbm.load_state_dict(torch.load(input_rbm))
    # ais=AnnealedImportanceSampler(rbm=rbm, n_gibbs_sampling_steps=10)
    # ais.sample()
    n_visible=10
    n_hidden=10
    eps=ExactPartitionSolver(n_visible=n_visible,n_hidden=n_hidden)
    eps.loadFromFile()
    Z,logZ=eps.calculatePartitionFct()
    print("Exact Partition Fct for RBM({0},{1})".format(n_visible,n_hidden))
    print("Z={0}".format(Z))
    print("logZ={0}".format(logZ))

    import pickle
    f=open('/Users/drdre/codez/qVAE/DiVAE/output/210324_eps/rbm_1010.pkl','wb')
    pickle.dump(eps,f)
    pickle.dump(Z,f)
    pickle.dump(logZ,f)
    f.close()