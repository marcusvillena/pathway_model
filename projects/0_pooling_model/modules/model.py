import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Literal

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

class SequentialModel(nn.Module):
    '''
    class using get_layers to dynamically construct a sequential container.
    similar to nn.Sequential but compatible with PyG models.
    '''
    def __init__(self, in_channels:int, out_channels:int, layer_class, layer_kwargs:dict={}, hidden_dims:list=[], act_fn=nn.LeakyReLU(), end_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

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