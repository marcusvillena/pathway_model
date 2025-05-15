from .utils import reshape, filter_kwargs

import copy
import functools
import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from typing import Literal, Optional, Union

def cloneable(model):
    """
    Decorator: make a model cloneable for training
    """
    # save original model.__init__
    model_init = model.__init__

    # saves previous typehinting, etc.
    @functools.wraps(model_init) 

    # override __init__ to capture args, kwargs
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        model_init(self, *args, **kwargs) # run original model init

    # define cloning function
    def clone(self, with_state:bool=False, **override_kwargs):
        # initiate new model
        init_kwargs = {**self.init_kwargs, **override_kwargs}
        new_model = type(self)(*self.init_args, **init_kwargs)

        # update state dict (e.g. cloning trained models)
        if with_state and hasattr(self, 'state_dict'):
            new_model.load_state_dict(self.state_dict())

        return new_model
    
    # update model
    model.__init__ = __init__ # override init
    model.clone = clone # add clone function
    model.is_trained = False # add tracker

    return model

def get_layers(
    in_channels:int, 
    out_channels:int, 
    layer_class:nn.Module, 
    hidden_dims:list[int]=None, 
    act_fn:nn.Module=None, 
    norm_fn:Literal['batch','layer']=None, 
    end_fn:Union[bool,nn.Module]=False,
    layer_kwargs:dict={}
):
    '''
    * dynamically constructs a ModuleList of a given layer_class and act_fn. 
    * first two args of the layer_class must be (in_channels, out_channels).
    * act_fn can be module class or module (pre-init)
    * end_fn=True uses act_fn, or can use separate nn.Module; False no final.
    '''
    # init
    layers = nn.ModuleList()
    in_dim = in_channels # first in_dim is in_channels

    # defaults
    hidden_dims = [] if hidden_dims is None else hidden_dims
    act_fn = nn.ReLU if act_fn is None else act_fn
    
    # helper
    def add_fn(layers:nn.ModuleList, item:Union[nn.Module,tuple]):
        if isinstance(item, type) and issubclass(item, nn.Module): # initialize and append
            item = item()
            item.forward = filter_kwargs(item.forward)
            layers.append(item)
        elif isinstance(item, nn.Module): # deepcopy pre-initialized
            item = copy.deepcopy(item)
            item.forward = filter_kwargs(item.forward)
            layers.append(copy.deepcopy(item))
        else:
            raise TypeError(f'unsupported type: {type(item)}')

    # define hidden layers
    for hidden_dim in hidden_dims:
        # init class
        layer = layer_class(in_dim, hidden_dim, **layer_kwargs)
        layer.forward = filter_kwargs(layer.forward) # filter kwargs in forward
        layers.append(layer)

        # norm
        if norm_fn == 'batch':
            add_fn(layers, nn.BatchNorm1d(hidden_dim))
        elif norm_fn == 'layer':
            add_fn(layers, nn.LayerNorm(hidden_dim))

        # activation function
        add_fn(layers, act_fn)

        # set next in_dim as current hidden_dim
        in_dim = hidden_dim 

    # final layer
    layer = layer_class(in_dim, out_channels, **layer_kwargs)
    layer.forward = filter_kwargs(layer.forward) # filter kwargs in forward
    layers.append(layer)

    # end fn
    if end_fn is True:
        add_fn(layers, act_fn) # true = use act_fn
    elif end_fn is not False:
        add_fn(layers, end_fn) # custom end_fn

    return layers

def attn_dims(embed_dim:Optional[int]=None, head_dim:Optional[int]=None, num_heads:int=1):
    # none specified; assert error
    assert (embed_dim is not None) or (head_dim is not None), 'one of [embed_dim, head_dim] must be specified'

    # both specified; lin_out reshapes head to embed
    if (embed_dim is not None) and (head_dim is not None):
        assert embed_dim // num_heads == head_dim, 'transformer dims incompatible, (embed_dim // num_heads == head_dim) must be true'
        return embed_dim, head_dim, num_heads

    # embed_dim specified; head = embed / num_heads
    elif embed_dim is not None:
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        head_dim = embed_dim // num_heads

    # head_dim specified; embed = head * num_heads
    elif head_dim is not None:
        embed_dim = head_dim * num_heads

    return embed_dim, head_dim, num_heads

@cloneable
class SequentialModel(nn.Module):
    '''
    class using get_layers to dynamically construct a sequential container.
    similar to nn.Sequential but compatible with PyG models.
    '''
    def __init__(
        self,
        in_channels:int, 
        out_channels:int, 
        layer_class:nn.Module, 
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,
        layer_kwargs:dict={},
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # define layers
        self.layers = get_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_class=layer_class,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn,
            layer_kwargs=layer_kwargs
        )
    
    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        return x

@cloneable
class NBGLM(nn.Module):
    def __init__(self, in_features:int, out_features:int, init_mu:float=8.5, init_theta:float=2.5, eps:float=1e-8, use_covariates:bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            x = reshape(x=x, to='b,n*f', num_nodes=self.out_features, num_node_features=self.in_features)
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