                    # print("h_sampled[0] {:.15f}".format(h_sampled[0][0]))
"""
Annealed Importance Sampling for approximating the free energy of an RBM.

Adapted from: https://pydeep.readthedocs.io/en/latest/_modules/pydeep/rbm/estimator.html?highlight=pydeep.rbm.estimator
"""

import numpy as np
import torch
from torch import nn
# torch.manual_seed(-123)
from copy import deepcopy

from models.rbm.rbm import RBM
from models.rbm.rbmBase import RBMBase
from models.samplers.baseSampler import BaseSampler

from DiVAE import logging
logger = logging.getLogger(__name__)

class AnnealedImportanceSampler(BaseSampler):
    def __init__(self, target_rbm=None, n_betas=10000, n_ais_chains=100, **kwargs):
        super(AnnealedImportanceSampler, self).__init__(**kwargs)

        #List of n_beta evenly spaced numbers over [0,1] - inverse temps
        self._betas=torch.linspace(0.0,1.0,n_betas)

        #this is necessary as the sampling procedure is sensitive to floating
        #number precision
        self.dtype=torch.float64
        
        self._out_fct=nn.Sigmoid()

        #number of simultaneous AIS chains
        self._n_ais_chains = n_ais_chains

        #the target rbm
        self._target_rbm=target_rbm
        #make sure to use double precision
        self._target_rbm.weights=self._target_rbm.weights.type(self.dtype)
        self._target_rbm.visible_bias=self._target_rbm.visible_bias.type(self.dtype)
        self._target_rbm.hidden_bias=self._target_rbm.hidden_bias.type(self.dtype)

        #the proposal distribution/base-rate RBM
        self._base_rbm = self.create_base_model()


    def create_base_model(self):
        #This base rate RBM A is the starting point of the annealing schedule.
        #base RBM: A, target RBM: B
        #parameter settings:
        #visible bias a_A=a_B
        #hidden bias b_A=b_B
        #weights=0
        rbm=RBMBase(n_visible=self._target_rbm.n_visible,n_hidden=self._target_rbm.n_hidden)
        rbm.weights = torch.zeros(self._target_rbm.weights.size())
        rbm.weights=rbm.weights.type(self.dtype)

        rbm.visible_bias = self._target_rbm.visible_bias
        #small number #TODO
        # tmp=torch.zeros(self._target_rbm.visible_bias.size())+0.00001
        # visb=torch.log(tmp)-torch.log(1-tmp)
        # rbm.visible_bias = visb.type(self.dtype)
        # print(rbm.visible_bias)
        # exit()
        return rbm

    def unnorm_log_prob_v(self,v,beta=None):
        activation = torch.matmul(v, self._target_rbm.weights.T) + self._target_rbm.hidden_bias
        bias = torch.matmul(v, self._target_rbm.visible_bias.T).reshape(v.shape[0], 1)

        if beta is not None:
            activation *= beta
            bias *= beta
            bias += (1.0 - beta) * torch.matmul(v, self._base_rbm.visible_bias.T).reshape(v.shape[0], 1)

        return bias + torch.sum(torch.log(torch.exp(activation) + 
            torch.exp(torch.zeros(activation.size()))), axis=1).reshape(v.shape[0], 1)

    def unnorm_log_prob_v_old(self, v, beta=None, printme=False):
        lnpv_0=torch.sum((1-beta)*self._base_rbm.visible_bias*v,1)
        # act_A=self._base_rbm.hidden_bias
        lnpv_1=torch.sum(torch.log(1+torch.exp((1-beta)*act_A)))

        #beta*bB*v
        lnpv_2=beta*torch.sum(self._target_rbm.visible_bias*v,1)

        act_B=torch.matmul(v,self._target_rbm.weights.T)+self._target_rbm.hidden_bias
        
        #sum hid(log(1+exp(beta*(wB*v+aB)))
        lnpv_3=torch.sum(torch.log(1+torch.exp(beta*act_B)),1)

        lnpv=lnpv_0+lnpv_1+lnpv_2+lnpv_3
        # lnpv=lnpv_0+lnpv_2+lnpv_3
        return lnpv.detach()

    def get_logZ_base_model(self):
        logZ=self._target_rbm._n_hidden*np.log(2)+torch.sum(torch.log(1+torch.exp(self._base_rbm.visible_bias)))
        return logZ

    def sample(self):
        torch.manual_seed(1)

        ############# 1.
        # The base model is the RBM with 0 weights, only the biases contribute to
        # the partition fct.
        # Sample the first time from the base model
        # base_rbm=self.create_base_model()
        #p(v|h) base model = sig(vis bias)
        #because Wij=0
        dummy=torch.zeros(self._n_ais_chains,self._base_rbm.visible_bias.size()[0])+self._base_rbm.visible_bias
        p_v_init=self._out_fct(dummy)
        #this size is determined by the number of AIS chains M we want to run
        #(see DBN paper "On the Quantitative Analysis of Deep Belief Networks")
        #and the size of the RBM visible layer - we want M v^(i) samples and
        #each v^(i) = n_visible nodes of RBM
        # v_size=(self._n_ais_chains,p_v_init.size()[0])
        rnd_nr=torch.rand(p_v_init.size())
        #initial sampling step: create M v_0 from base distribution RBM_A  
        v_sampled=torch.where(p_v_init>rnd_nr,1.,0.)

        # v_sampled

        # v_sampled=torch.zeros(v_sampled.size())
        ############# 2.
        #initial base ratex pA: 
        #negative as this is the divisor of the first factor in the w_AIS calc
        # act_base=torch.sum(self._base_rbm.visible_bias*v_sampled,1)
        # zA = self._n_hidden*np.log(2)+torch.sum(torch.log(1+torch.exp(self._base_rbm.hidden_bias)))
        # logpvsum=-(act_base+zA)

        #initial base rate pA: 
        logpvsum=-self.unnorm_log_prob_v(v=v_sampled,beta=0)

        ############# 3.
        betas=self._betas[1:self._betas.shape[0] - 1]
        for beta in betas:
            logpvsum+=self.unnorm_log_prob_v(v_sampled, beta=beta)
            ############# 4
            # Gibbs Sampler
            for step in range(self._n_gibbs_sampling_steps):
                #sample h 
                hid_act=beta * (torch.matmul(v_sampled,self._target_rbm.weights.T)+self._target_rbm.hidden_bias)
                phv=self._out_fct(hid_act)
                rnd_nr=torch.rand(phv.size()) 

                h_sampled=torch.where(phv>rnd_nr,1.,0.)
                vis_act=(1-beta)*self._base_rbm.visible_bias+beta*self._target_rbm.visible_bias+beta*torch.matmul(h_sampled,self._target_rbm.weights.T)

                pvh=self._out_fct(vis_act)
                rnd_nr=torch.rand(pvh.size()) 
                v_sampled=torch.where(pvh>rnd_nr,1.,0.)

            logpvsum-=self.unnorm_log_prob_v(v_sampled, beta=beta)

        beta_K=self._betas[len(self._betas)-1]
        logpvsum+=self.unnorm_log_prob_v(v_sampled, beta=beta_K)
        logz=torch.logsumexp(logpvsum,0)-np.log(self._n_ais_chains)
        baselogz = self.get_logZ_base_model()
        logz = logz + baselogz
        # print("{:.15f}".format(logz[0]))
        return logz


if __name__=="__main__": 
    logger.info("Loading Model")
    # rbm=RBM(n_visible=200,n_hidden=200)
    # input_rbm="/Users/drdre/codez/qVAE/DiVAE/outputs/2021-03-17/10-54-18/rbm.pt"
    # rbm.load_state_dict(torch.load(input_rbm))
    from models.rbm.exactRBMPartFctSolver import ExactPartitionSolver
    import pickle
    f=open("/Users/drdre/codez/qVAE/DiVAE/output/210324_eps/rbm_1010.pkl",'rb')
    # Exact Partition Fct for RBM(10,10)
    # Z=4886438400.0
    # logZ=22.309728622436523
    eps=pickle.load(f)
    Z=pickle.load(f)
    logZ=pickle.load(f)
    n_visible=eps._rbm_visible_bias.size()[0]
    n_hidden=eps._rbm_hidden_bias.size()[0]
    rbm=RBM(n_visible=n_visible,n_hidden=n_hidden)

    rbm._weights=torch.nn.Parameter(eps._rbm_weights,requires_grad=False)
    rbm._visible_bias=torch.nn.Parameter(eps._rbm_visible_bias,requires_grad=False)
    rbm._hidden_bias=torch.nn.Parameter(eps._rbm_hidden_bias,requires_grad=False)
    
    ais=AnnealedImportanceSampler(rbm=rbm, n_gibbs_sampling_steps=1)
    ais.sample()