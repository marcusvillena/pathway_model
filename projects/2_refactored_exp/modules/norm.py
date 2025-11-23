from .utils import reshape, cloneable
import torch
import torch.nn as nn

#typing
from .train import Loader
from torch import Tensor

@cloneable
class Normalizer(nn.Module): 
    # parent class template, change in child
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_with_loader(self, loader:Loader):
        pass

    def transform(self, x):
        return x
    
    def revert(self, x):
        return x

@cloneable
class logNorm(Normalizer):
    def transform(self, x):
        return torch.log1p(x)

    def revert(self, x):
        return torch.expm1(x)
    
@cloneable
class ZlogNorm(Normalizer):
    def init_with_loader(self, loader:Loader):
        self.num_nodes = loader.dataset[0].num_nodes
        self.num_features = loader.dataset[0].num_features
        self.log_mean = loader.stats['log_mean']
        self.log_std = loader.stats['log_std']

    def transform(self, x:Tensor):
        # get original shape
        orig_shape = x.shape

        # reshape for calc
        x = reshape(x, 'b,n,f', num_nodes=self.num_nodes, num_features=self.num_features)

        # get log_x
        log_x = torch.log1p(x)

        # get z from log_x
        z = (log_x - self.log_mean)/self.log_std

        # return in orig shape
        return z.reshape(orig_shape)
    
    def revert(self, z:Tensor):
        # get original shape
        orig_shape = z.shape

        # reshape for calc
        z = reshape(z, 'b,n,f', num_nodes=self.num_nodes, num_features=self.num_features)

        # get log_x from z
        log_x = (z * self.log_std) + self.log_mean

        # exp to get x
        x = torch.expm1(log_x)

        # return in orig shape
        return x.reshape(orig_shape)