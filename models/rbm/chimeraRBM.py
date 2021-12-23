"""
PyTorch implementation of a restricted Boltzmann machine with a Chimera topology
"""
import numpy as np
import torch
import math

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

from models.rbm.rbm import RBM

from DiVAE import logging
logger = logging.getLogger(__name__)

_CELL_SIDE_QUBITS = 4
_MAX_ROW_COLS = 16

class ChimeraRBM(RBM):
    def __init__(self, n_visible, n_hidden, **kwargs):
        super(ChimeraRBM, self).__init__(n_visible, n_hidden, **kwargs)

        self._n_visible=n_visible
        self._n_hidden=n_hidden
        
        # random weights and biases for all layers
        # weights between visible and hidden nodes. 784x128 (that is 28x28 input
        #size, 128 arbitrary choice)
        # if requires_grad=False : we calculate the weight update ourselves, not
        # through backpropagation
        require_grad=True
        
        n_cells = max(math.ceil(n_visible/_CELL_SIDE_QUBITS), math.ceil(n_hidden/_CELL_SIDE_QUBITS))
        n_rows = math.ceil(math.sqrt(n_cells))
        n_cols = n_rows
        
        assert n_cols<=_MAX_ROW_COLS
        
        visible_qubit_idxs = []
        hidden_qubit_idxs = []
        edge_list = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                for n in range(_CELL_SIDE_QUBITS):
                    if len(visible_qubit_idxs) < n_visible:
                        idx = 8*row + 8*col*_MAX_ROW_COLS + n
                        # Even cell
                        if (row+col)%2 == 0:
                            visible_qubit_idxs.append(idx)
                            hidden_qubit_idxs.append(idx+4)
                        # Odd cell
                        else:
                            hidden_qubit_idxs.append(idx)
                            visible_qubit_idxs.append(idx+4)
                        
                        # Add inner couplings within a K4,4 cell
                        for m in range(_CELL_SIDE_QUBITS):
                            opp_idx = 8*row + 8*col*_MAX_ROW_COLS + 4 + m
                            edge_list.append((idx, opp_idx))
                        
                        # Add external couplings
                        # Horizontal couplings
                        if (col+1) < n_cols:
                            end_idx = 8*row + 8*(col+1)*_MAX_ROW_COLS + n
                            edge_list.append((idx, end_idx))
                    
                        # Vertical couplings
                        if (row+1) < n_rows:
                            end_idx = 8*(row+1) + 8*col*_MAX_ROW_COLS + n
                            edge_list.append((idx+4, end_idx+4))
                            
        # Prune the edgelist to remove couplings between qubits not in the RBM
        pruned_edge_list = []
        for edge in edge_list:
            # Coupling between RBM qubits
            if (edge[0] in visible_qubit_idxs and edge[1] in hidden_qubit_idxs) or (edge[0] in hidden_qubit_idxs and edge[1] in visible_qubit_idxs):
                pruned_edge_list.append(edge)

        logger.debug("left = ", visible_qubit_idxs)
        logger.debug("right = ", hidden_qubit_idxs)
        logger.debug("edge_list = ", pruned_edge_list)
        
        # Chimera-RBM matrix
        visible_qubit_idx_map = {visible_qubit_idx:i for i, visible_qubit_idx in enumerate(visible_qubit_idxs)}
        hidden_qubit_idx_map = {hidden_qubit_idx:i for i, hidden_qubit_idx in enumerate(hidden_qubit_idxs)}
        
        weights_mask = torch.zeros(n_visible, n_hidden, requires_grad=False)
        for edge in pruned_edge_list:
            if edge[0] in visible_qubit_idxs:
                weights_mask[visible_qubit_idx_map[edge[0]], hidden_qubit_idx_map[edge[1]]] = 1.
            else:
                weights_mask[visible_qubit_idx_map[edge[1]], hidden_qubit_idx_map[edge[0]]] = 1.
        logger.debug("weights_mask = ", weights_mask)
                        
        #arbitrarily scaled by 0.01 
        self._weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01, requires_grad=require_grad)
        self._weights_mask = nn.Parameter(weights_mask, requires_grad=False)
        
        # Turn off RBM for one exp.
        #self._weights = nn.Parameter(torch.randn(n_visible, n_hidden) * 0., requires_grad=False)
        # all biases initialised to 0.5
        self._visible_bias = nn.Parameter(torch.ones(n_visible) * 0.5, requires_grad=require_grad)
        # #applying a 0 bias to the hidden nodes
        self._hidden_bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=require_grad)
        
    @property
    def weights(self):
        return self._weights * self._weights_mask
    
    @property
    def visible_bias(self):
        return self._visible_bias
    
    @property
    def hidden_bias(self):
        return self._hidden_bias
        
if __name__=="__main__":
    logger.debug("Testing chimeraRBM")
    cRBM = ChimeraRBM(8, 8)
    logger.debug("Success")