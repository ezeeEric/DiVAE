"""
Annealed Importance Sampling for approximating the free energy of an RBM.

Adapted from: https://pydeep.readthedocs.io/en/latest/_modules/pydeep/rbm/estimator.html?highlight=pydeep.rbm.estimator
"""

import numpy as np
import torch
from torch import nn
from copy import deepcopy

from models.rbm.rbm import RBM
from models.rbm.rbmBase import RBMBase
from models.samplers.baseSampler import BaseSampler

from DiVAE import logging
logger = logging.getLogger(__name__)

class AnnealedImportanceSampler(BaseSampler):
    def __init__(self, target_rbm=None, n_betas=10000, n_ais_chains=1000, **kwargs):
        super(AnnealedImportanceSampler, self).__init__(**kwargs)
        #number of inverse temperatures
        self._n_betas = n_betas

        #number of simultaneous AIS chains
        self._n_ais_chains = n_ais_chains

        #List of n_beta evenly spaced numbers over [0,1] - inverse temps
        self._betas=np.linspace(0.0,1.0,self._n_betas)
    
        #the target
        self._target_rbm=target_rbm

        self._n_visible = self._target_rbm._n_visible
        self._n_hidden  = self._target_rbm._n_hidden

        #the proposal distribution/base-rate RBM
        self._base_rbm = self.create_base_model()


    def create_base_model(self):
        #This base rate RBM A is the starting point of the annealing schedule.
        #base RBM: A, target RBM: B
        #parameter settings:
        #visible bias a_A=a_B
        #hidden bias b_A=b_B
        #weights=0
        rbm=RBMBase(n_visible=self._n_visible,n_hidden=self._n_hidden)
        rbm.weights = torch.zeros(self._target_rbm.weights.size())
        rbm.visible_bias = self._target_rbm.visible_bias
        rbm.hidden_bias = self._target_rbm.hidden_bias

        print(self._target_rbm.weights)
        print(rbm.visible_bias)
        print(rbm.hidden_bias)
        return rbm

    def unnorm_log_prob_v(self, v, beta=None):
        #(1-beta)*bA*v
        # print(v)
        # print("b_A:\t",self._base_rbm.visible_bias)
        # print("a_A:\t",self._base_rbm.hidden_bias)
        # print("b_B:\t",self._target_rbm.visible_bias)
        # print("a_B:\t",self._target_rbm.hidden_bias)
        # print("w_B:\t",self._target_rbm.weights)

        lnpv_0=torch.sum((1-beta)*self._base_rbm.visible_bias*v,1)
        #A_A=W_A*v+a_A=a_A as W_A=0
        act_A=self._base_rbm.hidden_bias
        lnpv_1=torch.sum(torch.log(1+torch.exp((1-beta)*act_A)))
        
        #beta*bB*v
        lnpv_2=beta*torch.sum(self._target_rbm.visible_bias*v,1)
        act_B=torch.matmul(v,self._target_rbm.weights.T)+self._target_rbm.hidden_bias
        
        #sum hid(log(1+exp(beta*(wB*v+aB)))
        lnpv_3=torch.sum(torch.log(1+torch.exp(beta*act_B)),1)
        lnpv=lnpv_0+lnpv_1+lnpv_2+lnpv_3
        return lnpv.detach()
    #https://pydeep.readthedocs.io/en/latest/_modules/pydeep/rbm/model.html
    # def unnormalized_log_probability_v(self, v, beta=None):
    #     #v*w+b
    #     activation = numx.dot(v, self.w) + self.bh
    #     #bias=v*b
    #     bias = numx.dot(temp_v, self.bv.T).reshape(temp_v.shape[0], 1)
    #     if beta is not None:
    #         #init-model: beta=0
    #         activation *= beta
    #         bias *= beta
    #         #init call
    #         if use_base_model is True:
    #             #bias=v*a_init
    #             bias += (1.0 - beta) * numx.dot(temp_v, self.bv_base.T).reshape(temp_v.shape[0], 1)
    #     return bias + 
    #     numx.sum(numx.log(numx.exp(activation * (1.0 - self.oh)) 
    #     + numx.exp(-activation * self.oh)), axis=1).reshape(v.shape[0], 1)

    def get_logZ_base_model(self):
        #M*log 2+sum(log(1+exp(vis_b)))
        logZ=self._n_hidden*np.log(2)+torch.sum(torch.log(1+torch.exp(self._base_rbm.visible_bias)))
        return logZ

    def sample(self):
        ############# 1.
        # The base model is the RBM with 0 weights, only the biases contribute to
        # the partition fct.
        # Sample the first time from the base model
        out_fct=nn.Sigmoid()
        base_rbm=self.create_base_model()
        #p(v|h) base model = sig(vis bias)
        #because Wij=0
        p_v_init=out_fct(self._base_rbm.visible_bias)
        #this size is determined by the number of AIS chains M we want to run
        #(see DBN paper "On the Quantitative Analysis of Deep Belief Networks")
        #and the size of the RBM visible layer - we want M v^(i) samples and
        #each v^(i) = n_visible nodes of RBM
        v_size=(self._n_ais_chains,p_v_init.size()[0])
        rnd_nr=torch.rand(v_size) 
        #initial sampling step: create M v_0 from base distribution RBM_A  
        v_sampled=torch.where(rnd_nr>=p_v_init,1.,0.)
        ############# 2.
        #initial base ratex pA: 
        #negative as this is the divisor of the first factor in the w_AIS calc
        # act_base=torch.sum(self._base_rbm.visible_bias*v_sampled,1)
        # zA = self._n_hidden*np.log(2)+torch.sum(torch.log(1+torch.exp(self._base_rbm.hidden_bias)))
        # logpvsum=-(act_base+zA)

        #initial base rate pA: 
        logpvsum=-self.unnorm_log_prob_v(v=v_sampled,beta=0)
        ############# 3.
        for beta in self._betas[1:][:-1]:
            # print(beta)
            logpvsum+=self.unnorm_log_prob_v(v_sampled, beta=beta)

            ############# 4
            # Gibbs Sampler
            for step in range(self._n_gibbs_sampling_steps):
                #sample h 
                hid_act=beta * (torch.matmul(v_sampled,self._target_rbm.weights.T)+self._target_rbm.hidden_bias)
                # print(hid_act.size())
                # hid_act=hid_act.reshape(self._n_ais_chains,1)
                phv=out_fct(hid_act)
                # phv=torch.exp(hid_act)/(1+torch.exp(hi))
                rnd_nr=torch.rand(phv.size()) 
                h_sampled=torch.where(rnd_nr>=phv,1.,0.)

                # b_vis=1/(1+np.exp(-(1-beta))*self._base_rbm.visible_bias+beta*self._target_rbm.visible_bias)
                # b_vis=(1-beta)*self._base_rbm.visible_bias+beta*self._target_rbm.visible_bias
                # vis_act=beta*torch.matmul(h_sampled,self._target_rbm.weights.T)+b_vis
                vis_act=1/(1+np.exp(-(1-beta))*self._base_rbm.visible_bias-beta*self._target_rbm.visible_bias-beta*torch.matmul(h_sampled,self._target_rbm.weights.T))
                # pvh=out_fct(vis_act)
                # rnd_nr=torch.rand(pvh.size()) 
                rnd_nr=torch.rand(vis_act.size()) 
                v_sampled=torch.where(rnd_nr>=vis_act,1.,0.)
                # v_sampled=torch.where(rnd_nr>=pvh,1.,0.)
            
            logpvsum-=self.unnorm_log_prob_v(v_sampled, beta=beta)
        beta_K=self._betas[len(self._betas)-1]
        logpvsum+=self.unnorm_log_prob_v(v_sampled, beta=beta_K)

        #combine all simultaneous AIS chains
        #TODO check this
        logz=torch.logsumexp(logpvsum,0)-np.log(self._n_ais_chains)
        baselogz = self.get_logZ_base_model()
        # Add the base partition function
        print(logz)
        print(baselogz)
        logz = logz + baselogz


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
    # n=2
    # x=int(2**n)
    # print(binary(torch.IntTensor(x),n))
    # # print(bin(x))
