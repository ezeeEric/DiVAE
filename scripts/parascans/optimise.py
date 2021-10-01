import os
import pickle
import torch
import io
from coffea import hist
import numpy as np
import matplotlib.pyplot as plt
import torch
# import pandas as pd

class Metrics(object):
    def __init__(self, idx=-1, allMetrics={}, cfgDict={}):
        
        self.idx=idx
        self._allMetrics=allMetrics
        self.cfgDict=cfgDict

        self.lossMetrics={}
        self.histogramMetrics={}
        
        self.sortMetrics()

    def sortMetrics(self):
        # print(self._allMetrics)
        # exit()
        for key,val in self._allMetrics.items():
            if isinstance(val,dict):
                newval={}
                for k2, v2 in val.items():
                    if isinstance(v2,tuple):
                        newval["_".join([key,k2,"stat"])]=v2[0]
                        newval["_".join([key,k2,"pval"])]=v2[1]
                    else:    
                        newval["_".join([key,k2])]=v2
                self.histogramMetrics=dict(**self.histogramMetrics,**newval)
            else:
                self.lossMetrics[key]=val.item() if isinstance(val,torch.Tensor) else val
                self.lossMetrics[key]=abs(self.lossMetrics[key])
        # statistic, pval

    def printCfg(self):
        for key,val in self.cfgDict.items():
            print("{0}: {1}".format(key,val))

    def __getattr__(self,item):
        try:
            return self.__dict__[item]
        except:
            metricDict=self.lossMetrics if item in self.lossMetrics.keys() else self.histogramMetrics
            return metricDict[item]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def readPickles(indir=""):
    summaryList=[]
    for root,dirs,files in os.walk(indir):
        for f_it in files:
            if not f_it.endswith("pkl"): continue
            fullFile=os.path.join(indir,f_it)
            f=open(fullFile,"rb")
            sumDict=CPU_Unpickler(f).load()
            cfgDict=CPU_Unpickler(f).load()
            metric=Metrics(idx=int(f_it.split("--")[0]),allMetrics=sumDict, cfgDict=cfgDict) 
            summaryList.append(metric)
            f.close()

    summaryList.sort(key=lambda x: x.idx)
    return summaryList
 
def plotMetricSpread(inlist=[], key="",outpath=""):
    minBin=getattr(min(inlist, key=lambda x: getattr(x,key)),key)
    maxBin=getattr(max(inlist, key=lambda x: getattr(x,key)),key)

    c_hist = hist.Hist(label="Events",
                        axes=(hist.Cat("metric", ""),
                        hist.Bin("val", key,100,minBin,maxBin)))
    
    data=[getattr(x,key) for x in inlist]
    c_hist.fill(metric="validation", val=np.array(data))
    ax_0, ax_1 = c_hist.axes()[0], c_hist.axes()[1]
    
    if isinstance(ax_0, hist.Cat) and isinstance(ax_1, hist.Bin):
        cat_ax = ax_0
        bin_ax = ax_1
    elif isinstance(ax_0, hist.Bin) and isinstance(ax_0, hist.Cat):
        bin_ax = ax_0
        cat_ax = ax_1
    else:
        raise ValueError("Expected categorical and bin axis")
    
    cat_names = [identifier.name for identifier in cat_ax.identifiers()]
    bins = [ax_bin.mid for ax_bin in bin_ax.identifiers()]
    
    value_dict = {cat_name:c_hist.values(overflow='all')[(cat_name,)] for cat_name in cat_names}
    bins = [bins[0]] + bins + [bins[len(bins)-1]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cat_name in cat_names:
        ax.step(bins, value_dict[cat_name], label=cat_name)
        
    ax.legend(title=cat_ax.label, prop={'size': 10})
    ax.set_xlabel(bin_ax.label, fontsize='15')
    ax.set_ylabel(c_hist.label, fontsize='15')
    ax.tick_params(axis='both', which='major', labelsize=15)

    # if scale == 'log':
    #     ax.set_xscale('log')
        
    # ax.set_yscale('log')
    # ax.set_ylim(bottom=1)
    
    plt.savefig("{0}/abs_{1}.png".format(outpath,key), format='png')
    plt.close()    
    return

def getBestModel(inlist,key):

    minMetric=min(inlist, key=lambda x: getattr(x,key))
    maxMetric=max(inlist, key=lambda x: getattr(x,key))
    print("{0} - Min: {1} - Max: {2}".format(key, minMetric.idx, maxMetric.idx))

def addMetrics(x,keyList):
    sumMetrics=0
    for key in keyList:
        sumMetrics+=float(getattr(x,key))
    return sumMetrics

def getBestCombinedModel(inlist,keyList,ksKeys):

    def sumKSMetrics(x):
        return round(sum(list(map(lambda key: getattr(x,key),ksKeys))),1)
    
    def sumLossMetrics(x):
        return round(sum(list(map(lambda key: getattr(x,key),keyList))),1)
    
    # sortedMetrics=sorted(inlist, key=sumLossMetrics, reverse=False)
    sortedMetrics=sorted(inlist, key=sumKSMetrics, reverse=False)
    for result in ["{0}: ({1},{2},{3:.1f})".format(x.idx,sumKSMetrics(x),sumLossMetrics(x),x.lossMetrics["val_kl_loss"]) for x in sortedMetrics[0:10]]:
        print(result)
    
    # sortedMetrics=sorted(sortedMetrics, key=sumLossMetrics, reverse=False)
    # print([(x.idx,(sumLossMetrics(x))) for x in sortedMetrics[0:10]])

    # print([x.idx for x in sortedMetrics])
    # print([(x.idx,x.lossMetrics['val_kl_loss'],x.lossMetrics['val_ae_loss'],x.lossMetrics['val_hit_loss']) for x in sortedMetrics])
    # for i in range(4):
    #     bestRunIdz.append(sortedMetrics[-i].idx)
    # return bestRunIdz

def prefilterList(inlist, allkeys=["layer_0_EnergyHist_kstest_pval"], threshold=0.15, nrAllowedFails=2):
    #this function takes a bunch of histogram metrics, like the ksttest p-values
    #for various output histograms and loops over all runs. If a single run has
    #more than 2 histograms failing the requirement, it is rejected.
    keyList=[]
    for el in inlist:
        keyList=[key for key in allkeys if getattr(el,key)<threshold]
        if len(keyList)<nrAllowedFails:
            yield el

def analyseRuns(inlist=[], keywords=["ae_loss","hit_loss","kl_loss"], ksKeys=[],outpath=""):
    #prefilter the input list to only have situation where the kstest p-value is
    #greater than some threshold for all histograms
    # filterList=list(prefilterList(inlist=inlist,allkeys=prefilterKeys, threshold=0.25, nrAllowedFails=1))
    # print(len(filterList))
    # # print([x.idx for x in filterList])
    # for key in keywords:
    #     plotMetricSpread(inlist,key,outpath=outpath)
        # getBestModel(inlist,key)
    bestRuns=getBestCombinedModel(filterList,keyList=keywords,ksKeys=ksKeys)
    # print(bestRuns)
    # getCfgValues(inlist,bestRuns)
    return 

def getCfgValues(inlist=[],bestRuns=[]):
    for idx in bestRuns:
        metric=inlist[idx]
        print("\n{0}\n***********".format(idx))
        for key, val in dict(**metric.cfgDict['model'],**metric.cfgDict['engine']).items():
            print("{0} : {1}".format(key,val))

def printSpecificRuns(inlist=None,nrs=[]):
    for nr in nrs:
        thisrun=list(filter(lambda el: el.idx==nr,inlist))[0]
        # print(thisrun.idx)
        # print(thisrun.lossMetrics)
        # for key, val in thisrun.histogramMetrics.items():?
        print("{0}: ({1},{2})".format(thisrun.idx,thisrun.lossMetrics["val_kl_loss"],thisrun.lossMetrics["val_ae_loss"]))


if __name__=="__main__":
    # PTYPES=["gamma","eplus","piplus"]
    PTYPE="piplus"
    RAWINPATH="/Users/drdre/outputz/210923_parameterScan/{0}/runSummaries".format(PTYPE)
    VISPATH=os.path.join("/Users/drdre/outputz/210923_parameterScan/{0}/".format(PTYPE),"visualisations")
    
    os.system("mkdir -p {0}".format(VISPATH))

    summaryList=readPickles(indir=RAWINPATH)

    keys=["val_"+x for x in ["ae_loss",'kl_loss', 'hit_loss']]#, 'entropy', 'pos_energy', 'neg_energy','loss']]
    # keys=["val_kl_loss"]#+x for x in ["ae_loss",'kl_loss', 'hit_loss', 'entropy', 'pos_energy', 'neg_energy','loss']]

    # ksKeys=list(filter(lambda key: key.endswith('_kstest_pval'),summaryList[0].histogramMetrics.keys()))
    # ksKeys=list(filter(lambda key: key.endswith('layer_1_sparsityHist_kstest_pval') or key.endswith('layer_1_sparsityHist_kstest_pval'),summaryList[0].histogramMetrics.keys()))
    # ksKeys=list(filter(lambda key: key.endswith('layer_1_sparsityHist_dist') or key.endswith('layer_2_sparsityHist_dist'),summaryList[0].histogramMetrics.keys()))
    ksKeys=list(filter(lambda key: key.endswith('showerDepthHist_dist'),summaryList[0].histogramMetrics.keys()))
    # ksKeys=list(filter(lambda key: key.endswith('ist_dist'),summaryList[0].histogramMetrics.keys()))
    # keys=ksKeys
    # +ksKeys
    filterList=list(prefilterList(inlist=summaryList,allkeys=ksKeys, threshold=0.05, nrAllowedFails=2))
    # print(len(filterList))
    # for key in ksKeys:
    #     # print(key)
    #     analyseRuns(inlist=filterList, keywords=keys, ksKeys=[key] ,outpath=VISPATH)
    # analyseRuns(inlist=filterList, keywords=keys, ksKeys=ksKeys ,outpath=VISPATH)
    printSpecificRuns(summaryList,nrs=[143])
    # printSpecificRuns(inlist=summaryList, nrs=[41])
    # +distKeys
    #'ae_loss', 'kl_loss', 'hit_loss', 'entropy', 'pos_energy', 'neg_energy',
    #'gamma', 'epoch', 'loss' dict_keys(['totalEnergyHist_dist',
    #'totalEnergyHist_kstest_stat', 'totalEnergyHist_kstest_pval',
    #'totalEnergyHist_ttest_stat', 'totalEnergyHist_ttest_pval',
    #'layer_0_EnergyHist_dist', 'layer_0_EnergyHist_kstest_stat',
    #'layer_0_EnergyHist_kstest_pval', 'layer_0_EnergyHist_ttest_stat',
    #'layer_0_EnergyHist_ttest_pval', 'layer_0_fracEnergyHist_dist',
    #'layer_0_fracEnergyHist_kstest_stat', 'layer_0_fracEnergyHist_kstest_pval',
    #'layer_0_fracEnergyHist_ttest_stat', 'layer_0_fracEnergyHist_ttest_pval',
    #'layer_0_sparsityHist_dist', 'layer_0_sparsityHist_kstest_stat',
    #'layer_0_sparsityHist_kstest_pval', 'layer_0_sparsityHist_ttest_stat',
    #'layer_0_sparsityHist_ttest_pval', 'layer_1_EnergyHist_dist',
    #'layer_1_EnergyHist_kstest_stat', 'layer_1_EnergyHist_kstest_pval',
    #'layer_1_EnergyHist_ttest_stat', 'layer_1_EnergyHist_ttest_pval',
    #'layer_1_fracEnergyHist_dist', 'layer_1_fracEnergyHist_kstest_stat',
    #'layer_1_fracEnergyHist_kstest_pval', 'layer_1_fracEnergyHist_ttest_stat',
    #'layer_1_fracEnergyHist_ttest_pval', 'layer_1_sparsityHist_dist',
    #'layer_1_sparsityHist_kstest_stat', 'layer_1_sparsityHist_kstest_pval',
    #'layer_1_sparsityHist_ttest_stat', 'layer_1_sparsityHist_ttest_pval',
    #'layer_2_EnergyHist_dist', 'layer_2_EnergyHist_kstest_stat',
    #'layer_2_EnergyHist_kstest_pval', 'layer_2_EnergyHist_ttest_stat',
    #'layer_2_EnergyHist_ttest_pval', 'layer_2_fracEnergyHist_dist',
    #'layer_2_fracEnergyHist_kstest_stat', 'layer_2_fracEnergyHist_kstest_pval',
    #'layer_2_fracEnergyHist_ttest_stat', 'layer_2_fracEnergyHist_ttest_pval',
    #'layer_2_sparsityHist_dist', 'layer_2_sparsityHist_kstest_stat',
    #'layer_2_sparsityHist_kstest_pval', 'layer_2_sparsityHist_ttest_stat',
    #'layer_2_sparsityHist_ttest_pval', 'dwTotalEnergyHist_dist',
    #'dwTotalEnergyHist_kstest_stat', 'dwTotalEnergyHist_kstest_pval',
    #'dwTotalEnergyHist_ttest_stat', 'dwTotalEnergyHist_ttest_pval',
    #'showerDepthHist_dist', 'showerDepthHist_kstest_stat',
    #'showerDepthHist_kstest_pval', 'showerDepthHist_ttest_stat',
    #'showerDepthHist_ttest_pval'

# 30 TRAINPAR={
# 31         "engine.learning_rate":[0.0001,0.00005],
# 32         "engine.n_train_batch_size":[64],
# 33         "engine.n_epochs":[50],
# 34         #"engine.n_gibbs_sampling_steps":[20,40,60],
# 35         "engine.n_gibbs_sampling_steps":[40],
# 36         #"engine.kl_annealing_ratio":[0.1,0.3,0.5],
# 37         "engine.kl_annealing_ratio":[0.3],
# 38         }
# 39 
# 40 MODELPAR={
# 41     #    "activation_function":"relu",
# 42     #    "model.beta_smoothing_fct":[0.1,5,10],
# 43         "model.beta_smoothing_fct":[7],
# 44     #    "decoder_hidden_nodes":[[50,100,100],[200,300,400],[400,600,800]],
# 45     #    "encoder_hidden_nodes":[[100,100,50,],[400,300,200],[800,600,400]],
# 46         "model.n_encoder_layer_nodes":[128,400],
# 47         "model.n_encoder_layers":[2,4],
# 48         "model.n_latent_hierarchy_lvls": [2,4],
# 49         "model.n_latent_nodes": [128],
# 50         }