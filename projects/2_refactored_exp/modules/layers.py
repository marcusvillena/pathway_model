from .utils import attn_dims, cloneable, get_layers, filter_kwargs, input_to_dict

import torch
import torch.nn as nn

# typing
from .data import GraphDataset
from torch import Tensor
from torch_geometric.data import Data
from typing import Literal, Union


# general
@cloneable
class Dims():
    def __init__(
        self, 
        dataset:GraphDataset, 
        embed_dim:int=None, 
        head_dim:int=None, 
        num_heads:int=1, 
        method:Literal['node','set','twin']='node', 
        eps:float=1e-6
    ):
        # get datawrapper        
        self._dataset = dataset.wrapper
        self.mask = dataset.wrapper.pathway_index # legacy / compatibility
        self.num_sets = dataset.wrapper.num_pathways # legacy / compatibility
        self.embed_dim, self.head_dim, self.num_heads = attn_dims(embed_dim, head_dim, num_heads)
        self.eps = eps

        # for reshaping (mlp2, latent)
        if method == 'node':
            self.n_dim = self.num_nodes
            self.split_dim = None
        elif method == 'set':
            self.n_dim = self.num_sets
            self.split_dim = None
        elif method == 'twin':
            self.n_dim = self.num_nodes + self.num_sets
            self.split_dim = [self.num_nodes, self.num_sets]
        else:
            raise TypeError(f'unsupported method: {method}')
        
    def __getattr__(self, attr):
        # return _data attr if called
        return getattr(self._dataset, attr)
    
    def __dir__(self):
        # list _data attr in IDEs
        return list(self.__dict__.keys()) + dir(self._dataset)

@cloneable 
class Sequential(nn.Module):
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

        self.layers = get_layers(
            in_channels,
            out_channels,
            layer_class,
            hidden_dims,
            act_fn,
            norm_fn,
            end_fn,
            layer_kwargs
        )

    def forward(self, input:Union[Data, Tensor, dict], return_dict:bool=False, **kwargs) -> Union[dict,Tensor]:
        # get input as kwargs dict
        data = input_to_dict(input)

        # extract x as positional, update kwargs if provided
        x = data['x']
        data = {k:v for k,v in data.items() if k!='x'}
        data.update(kwargs)

        out = {}

        # forward pass through layers
        for layer in range(len(self.layers)):
            layer_out = filter_kwargs(self.layers[layer])(x, **data)

            # filter output if tuple
            if isinstance(layer_out, torch.Tensor):
                x = layer_out
                
            elif isinstance(layer_out, tuple):
                x = layer_out[0]
                remaining = layer_out[1:]

                if len(remaining) == 1: 
                    out[f'out_{layer}'] = remaining[0] # unpacks tuple if 1
                else:
                    out[f'out_{layer}'] = remaining # keeps tuple if 2+

            else:
                raise TypeError(f'unsupported layer output type: {type(layer_out)}')
            
        if return_dict:
            out['x'] = x
            return out
        else:
            return x

# pooling
@cloneable
class SetPooling(nn.Module):
    @filter_kwargs
    def __init__(self, mask:Tensor, num_features:int, *args, **kwargs):
        '''
        mask: (nodes, sets)
        should adapt to have edge features in future
        '''
        super().__init__()
        self.mask = mask
        self.num_nodes, self.num_sets = mask.shape
        self.num_features = num_features

    def forward(self, input:Union[Data, Tensor, dict], concat:bool=True, return_dict:bool=False):
        # get input as kwargs dict
        data = input_to_dict(input)

        # get x in (batch, nodes, features)
        x_node = data['x'].reshape(-1, self.num_nodes, self.num_features)

        # pool x to (batch, set, features)
        out = self.pool(x_node)
        x_set = out.get('x')

        # concat to (batch, nodes + sets, features)
        x = torch.cat([x_node, x_set], dim=1) if concat else x_set
        out['x'] = x

        return out if return_dict else out['x']

    def pool(self, x:Tensor):
        '''
        define in child class. default: mean
        '''
        # sum across sets (add pool)
        x_set = torch.einsum('bnf,ns->bsf', x, self.mask)

        # nodes per set (denom); clamp for sum=0 case
        nodes_per_set = self.mask.sum(dim=0).clamp(min=1).view(1,self.num_sets,1)

        # return mean (sum per set/total per set)
        mean = x_set / nodes_per_set

        return {'x':mean}
    
@cloneable
class AttentionSetPooling(SetPooling):
    def __init__(
        self, 
        mask:Tensor,
        num_features:int,

        # lin 
        hidden_dims:list[int]=None,
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        *args, **kwargs
    ):
        super().__init__(mask, num_features, *args, **kwargs)

        self.lin = Sequential(
            in_channels=num_features,
            out_channels=self.num_sets,
            layer_class=nn.Linear,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn
        )

    def pool(self, x:Tensor):
        # compute masked scores, attention
        scores = self.lin(x).masked_fill(self.mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=1) # dim 1 or -1?

        # apply attention (weighted mean)
        x = torch.einsum('bnf,bns->bsf', x, attn)

        return {'x':x, 'attn':attn}
    
@cloneable
class MultiheadSetPooling(SetPooling):
    def __init__(
        self, 
        mask:Tensor,
        num_features:int,
        dims:Dims, # for multihead
        aggregate:Literal['mean','concat']='concat',

        # lin 
        hidden_dims:list[int]=None,
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,
        *args,
        **kwargs,
    ):
        super().__init__(mask, num_features, *args, **kwargs)

        # note: head_dim not used, instead embed_dim = num_features
        # results in large in_channels for out_proj if concat
        self.aggregate = aggregate
        self.num_heads = dims.num_heads

        # layers
        self.lin = Sequential(
            in_channels=num_features,
            out_channels=self.num_sets * self.num_heads,
            layer_class=nn.Linear,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn
        )

        if self.aggregate == 'concat':
            self.out_proj = Sequential(
                in_channels=self.num_heads * num_features, # num_heads * embed_dim (large)
                out_channels=num_features,
                layer_class=nn.Linear,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )
        else:
            self.out_proj = None

    def pool(self, x:Tensor):
        # get scores
        scores = self.lin(x) # (b, n, s * h)
        scores = scores.view(-1, self.num_nodes, self.num_heads, self.num_sets) # (b, n, h, s)
        scores = scores.permute(0, 2, 1, 3) # (b, h, n, s)

        # mask scores, get attention
        mask = self.mask # (n, s)
        scores = scores.masked_fill(mask==0, float('-inf')) # (b, h, n, s)

        # attention over nodes
        attn = torch.softmax(scores, dim=2) # (b, h, n, s)
        
        # apply attention (weighted mean) over n: n,f * n,s -> s,f
        out = torch.einsum('bnf,bhns->bhsf', x, attn) # (b, h, s, f)

        # aggregate over h: (b,h,s,f) -> (b,s,f)
        if self.aggregate == 'mean':
            out = out.mean(dim=1)

        if self.aggregate == 'concat':
            out = out.permute(0,2,1,3) # (b,s,h,f)
            out = out.reshape(-1, self.num_sets, self.num_heads * self.num_features)
            out = self.out_proj(out)

        return {'x':out, 'attn':attn}