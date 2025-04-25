import torch
import torch.nn as nn
from torch import Tensor
from .model import SequentialModel

class SetPooling(nn.Module):
    def __init__(self, mask:Tensor, out_channels:int=None, *args, **kwargs):
        '''
        pools nodes into sets using weights generated from self.mask.
        mask: (num_nodes, num_sets)
        '''
        super().__init__(*args, **kwargs)

        # format to enable projection
        self.mask = mask.unsqueeze(0) # --> (1, num_nodes, num_sets)

    def forward(self, x:Tensor):
        '''
        forward pass, pooling x to z.
        x: (batch_size, num_nodes, num_features)
        z: (batch_size, num_sets, num_features)
        '''
        # compute weight
        self.weight = self._compute_weight(x) # (batch_size, num_sets, num_nodes)

        # pool (sum) into set: (b,s,n) @ (b,n,F)
        z = torch.bmm(self.weight, x) # (batch_size, num_sets, num_features)

        return z
    
    def _compute_weight(self, x:Tensor):
        '''
        computes weight from self.mask and x (if applicable). To be changed in child class
        '''
        pass

class AddSetPooling(SetPooling):
    def __init__(self, mask:Tensor, out_channels:int=None, *args, **kwargs):
        '''
        pools nodes into sets by addition
        mask: (num_nodes, num_sets)
        '''
        super().__init__(mask, out_channels, *args, **kwargs)
    
    def _compute_weight(self, x:Tensor):
        '''
        returns expanded, transposed mask
        '''
        batch_size = x.shape[0]

        # format mask as 'weight' for bmm
        weight = self.mask # --> (1, num_nodes, num_sets)
        weight = weight.transpose(1,2) # --> (1, num_sets, num_nodes)
        weight = weight.expand(batch_size, -1, -1) # --> (batch_size, num_sets, num_nodes)

        return weight

class WeightedSetPooling(SetPooling):
    def __init__(self, mask:Tensor, out_channels:int, hidden_dims:list=[], act_fn=nn.LeakyReLU(), end_fn=None, eps:float=1e-8, *args, **kwargs):
        '''
        non-normalized weighted set pooling.
        mask: (num_nodes, num_sets)
        '''
        self.eps = eps # small num to prevent division by 0
        super().__init__(mask, out_channels, *args, **kwargs)

        # define linear transform layer for learnable attention scores
        num_sets = mask.shape[1]
        
        # self.lin = nn.Linear(out_channels, num_sets)
        self.lin = SequentialModel(
            in_channels=out_channels,
            out_channels=num_sets,
            layer_class=nn.Linear,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            end_fn=end_fn
        )

    def _compute_weight(self, x:Tensor):
        # get scores via (b,n,F) @ (b,F,s) --> (batch_size, num_nodes, num_sets)
        scores = self.lin(x)

        # apply mask
        weight = scores * self.mask

        return weight

class SoftmaxSP(WeightedSetPooling):
    def _compute_weight(self, x):
        '''
        computes weight as softmax(scores); in [0,1] where sum = 1
        '''
        # get scores via (b,n,F) @ (b,F,s) --> (batch_size, num_nodes, num_sets)
        scores = self.lin(x)

        # apply mask
        scores = scores.masked_fill(self.mask == 0, float('-inf'))

        # softmax
        weight = torch.softmax(scores, dim=1)

        # transpose for bmm: (batch_size, num_sets, num_nodes)
        weight = weight.transpose(1, 2)

        return weight

class SoftTanhSP(WeightedSetPooling):
    def _compute_weight(self, x:Tensor):
        '''
        computes weight as tanh(scores)/L1norm(tanh(scores)); in [-1,1] where sum(abs) = 1
        '''
        # get scores via (b,n,F) @ (b,F,s); mask --> (batch_size, num_nodes, num_sets)
        scores = self.lin(x)
        scores = scores * self.mask

        # compute weight via tanh activation / L1norm
        scores = torch.tanh(scores)
        L1norm = torch.sum(torch.abs(scores), dim=1, keepdim=True)
        weight = scores / (L1norm + self.eps)
        weight = scores

        # transpose for bmm
        weight = weight.transpose(1,2) # --> (batch_size, num_sets, num_nodes)

        return weight