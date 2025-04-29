from .utils import reshape_x

import functools
import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from typing import Literal, Union

def get_layers(in_channels:int, out_channels:int, layer_class, layer_kwargs:dict={}, hidden_dims:list=[], act_fn=nn.LeakyReLU(), norm:Literal['pre','post']=None, end_fn=None):
    '''
    dynamically constructs a ModuleList of a given layer_class and act_fn. 
    first two args of the layer_class must be (in_channels, out_channels).
    '''
    layers = nn.ModuleList()
    in_dim = in_channels # first in_dim is in_channels

    for hidden_dim in hidden_dims:
        layers.append(layer_class(in_dim, hidden_dim, **layer_kwargs))
        
        if norm == 'pre':
            layers.append(nn.LayerNorm(hidden_dim))

        if act_fn != None:
            layers.append(act_fn)

        if norm == 'post':
            layers.append(nn.LayerNorm(hidden_dim))
            
        in_dim = hidden_dim # set next in_dim as current hidden_dim

    # final layer
    layers.append(layer_class(in_dim, out_channels, **layer_kwargs))

    # end_fn (if applicable)
    if end_fn != None:
        layers.append(end_fn)

    return layers

def cloneable(model):
    """
    Decorator: make a model cloneable for training
    """
    # save original model.__init__
    model_init = model.__init__

    @functools.wraps(model_init) # saves previous typehinting, etc.

    # override __init__ to capture args, kwargs
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        model_init(self, *args, **kwargs) # run original model init

    # define cloning function
    def clone(self, with_state=False, device=None, **override_kwargs):
        # initiate new model
        init_args = self._init_args
        init_kwargs = {**self._init_kwargs, **override_kwargs} # modify clone kwargs
        new_model = type(self)(*init_args, **init_kwargs)

        # update state dict (e.g. cloning trained models)
        if with_state and hasattr(self, 'state_dict'):
            new_model.load_state_dict(self.state_dict())

        # update device, if needed
        if device is not None:
            new_model = new_model.to(device)

        return new_model
    
    # update model
    model.__init__ = __init__ # override init
    model.clone = clone # add clone function
    model.is_trained = False # add tracker

    return model

@cloneable
class SequentialModel(nn.Module):
    '''
    class using get_layers to dynamically construct a sequential container.
    similar to nn.Sequential but compatible with PyG models.
    '''
    def __init__(self, in_channels:int, out_channels:int, layer_class, layer_kwargs:dict={}, hidden_dims:list=[], act_fn=nn.LeakyReLU(), end_fn=None, *args, **kwargs):
        super().__init__(in_channels, out_channels, layer_class, layer_kwargs, hidden_dims, act_fn, end_fn, *args, **kwargs)

        # define layers
        self.layers = get_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_class=layer_class,
            layer_kwargs=layer_kwargs,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            end_fn=end_fn
        )

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            if isinstance(layer, MessagePassing): # PyG handling
                x = layer(x, *args, **kwargs)
            else:
                x = layer(x) # normal PyTorch layer

        return x

@cloneable
class NBGLM(nn.Module):
    def __init__(self, in_features:int, out_features:int, init_mu:float=8.5, init_theta:float=2.5, eps:float=1e-8, use_covariates:bool=False, *args, **kwargs):
        super().__init__(in_features, out_features, init_mu, init_theta, eps, use_covariates, *args, **kwargs)
        self.eps = eps
        self.use_covariates = use_covariates
        self.in_features = in_features
        self.out_features = out_features

        # gene coeffs (beta) in (sample_features, num_nodes) - learned wt. per samp.
        if self.use_covariates:
            self.lin = nn.Linear(in_features, out_features)
        else:
            self.lin = nn.Linear(1, out_features)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.constant_(self.lin.bias, init_mu)

        # gene dispersions in (num_nodes,) - const. per gene
        self.log_theta = nn.Parameter(
            torch.full(
                size=(out_features,), 
                fill_value=torch.tensor(init_theta + self.eps).log()
            )
        )

    def forward(self, x:Union[Tensor, Batch, int], as_dict:bool=True, *args, **kwargs):
        '''
        Simple NB GLM for benchmarking. Passing nothing returns global estimates.

        x: design matrix in (batch_size, sample_features)
        mu: mean gene counts in (batch_size, num_nodes)
        theta: gene dispersion in (batch_size, num_nodes)
        '''
        # format x
        if isinstance(x, int):
            batch_size = x
        else:
            x = reshape_x(x=x, to='b,n*f', num_nodes=self.out_features, num_node_features=self.in_features)
            batch_size = x.shape[0]
        
        # create ones if global (no covariates)
        if not self.use_covariates:
            x = torch.ones(batch_size, 1)

        # estimate parameters
        mu = torch.exp(self.lin(x))
        theta = torch.exp(self.log_theta).expand(mu.shape)

        # predict x (mean predictor)
        x_recon = mu

        if as_dict:
            return {'x_recon':x_recon, 'mu':mu, 'theta':theta}
        else:
            return x_recon, mu, theta