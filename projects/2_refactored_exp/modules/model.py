from .layers import SetPooling, Sequential
from .utils import attn_dims, build_hidden_dims, cloneable, clone_or_init, input_to_dict, reshape
import torch
import torch.nn as nn

# typing
from .data import GraphDataset
from .norm import Normalizer
from torch import Tensor
from torch_geometric.data import Data
from typing import Literal, Union

## General

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
class Encoder(nn.Module):
    def __init__(
        self,
        dims:Dims,
        method:Literal['node','set']='node', # twin removed for now

        # layers
        norm_class:Normalizer=None,
        encoder_class:nn.Module=None,
        pooling_class:SetPooling=None,

        # new layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        # kwargs
        norm_kwargs:dict=None,
        encoder_kwargs:dict=None,
        pooling_kwargs:dict=None,

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs
        pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        self.method=method

        # dims
        self.mask = dims.mask
        self.embed_dim = dims.embed_dim
        self.num_nodes = dims.num_nodes
        self.num_node_features = dims.num_node_features
        hidden_dims = build_hidden_dims(self.embed_dim, hidden_dims)

        # init norm
        if norm_class is None:
            self.norm = Normalizer()
        else:
            self.norm = clone_or_init(
                name='norm_class',
                obj=norm_class,
                base_class=nn.Module,
                builder=lambda cls: cls(**norm_kwargs)
            )

        # init layers
        if encoder_class is None:
            encoder_class = nn.Linear
        self.node_encoder = clone_or_init(
            name='encoder_class',
            obj=encoder_class,
            base_class=nn.Module,
            builder=lambda cls: Sequential(
                in_channels=self.num_node_features,
                out_channels=self.embed_dim,
                layer_class=cls,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn,
                layer_kwargs=encoder_kwargs,
            )
        )

        # init pooling if method != 'node'
        if method == 'node':
            self.node_pooling = None
        else:
            self.node_pooling = clone_or_init(
                name='pooling_class',
                obj=pooling_class,
                base_class=SetPooling,
                builder = lambda cls: cls(
                    mask=self.mask,
                    num_features=self.embed_dim,
                    hidden_dims=hidden_dims,
                    act_fn=act_fn,
                    norm_fn=norm_fn,
                    end_fn=end_fn,
                    **pooling_kwargs
                )
            )

    def init_with_loader(self, loader): # pass loader to Encoder -> Norm
        if callable(getattr(self.norm, 'init_with_loader', None)):
            self.norm.init_with_loader(loader)

    def _normalize(self, data:dict):
        # ensure x is float
        data['x'] = data['x'].float() 

        # transform x to norm
        data['x'] = self.norm.transform(data['x'])

        return data

    def _encode(self, data:dict, need_weights:bool, **kwargs):
        # get node embedding
        ne_out = self.node_encoder(data, return_dict=need_weights, return_attention_weights=need_weights, **kwargs)

        # extract node embedding from output
        if isinstance(ne_out, Tensor):
            h_node = ne_out
        else:
            ne_out = input_to_dict(ne_out) # extract as dict
            h_node = ne_out.pop('x')

        # reshape to b,n,f (for pooling)
        h_node = reshape(h_node, 'b,n,f', num_nodes=self.num_nodes, num_features=self.embed_dim)

        return h_node, ne_out

    def _pool(self, h_node:Tensor, need_weights:bool):
        # no pooling
        if self.node_pooling is None:
            return h_node, {}
        
        # pooling
        else:
            # concat if method 'twin'
            concat = True if self.method == 'twin' else False

            # get pooled embedding
            np_out = self.node_pooling(h_node, concat=concat, return_dict=need_weights)

            # extract output
            if isinstance(np_out, Tensor):
                h_pool = np_out
            else:
                np_out = input_to_dict(np_out) # extract as dict
                h_pool = np_out.pop('x')

            return h_pool, np_out

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False, **kwargs):
        # extract x
        data = input_to_dict(input)

        # normalize
        data = self._normalize(data)

        # node embedding
        x, ne_out = self._encode(data, need_weights, **kwargs)
        h_node = x
        
        # node pooling
        x, np_out = self._pool(x, need_weights)
        h_pool = x if self.node_pooling is not None else None

        # format output
        out = {}
        out['x'] = x
        # out['mu'] = data.get('mu') # nbloss
        # out['theta'] = data.get('theta') # nbloss

        if need_weights:
            out['layer_outs'] = {}
            out['layer_outs']['ne'] = ne_out
            out['layer_outs']['np'] = np_out
            out['h_node'] = h_node
            out['h_pool'] = h_pool
        
        return out

@cloneable
class Latent(nn.Module):
    def __init__(
        self,
        dims:Dims,

        # layers
        mlp:Union[bool,nn.Module]=False,
        pooling_class:SetPooling=SetPooling,

        # new layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,   

        # kwargs
        mlp_kwargs:dict=None,
        pooling_kwargs:dict=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        mlp_kwargs = {} if mlp_kwargs is None else mlp_kwargs
        pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        
        # dims
        self.embed_dim = dims.embed_dim
        self.n_dim = dims.n_dim
        hidden_dims = build_hidden_dims(self.embed_dim, hidden_dims)

        # init mlp
        if mlp is False:
            self.mlp = None
        else:
            if mlp is True: # use default (nn.Linear)
                mlp = nn.Linear 
            self.mlp = clone_or_init(
                name='mlp',
                obj=mlp,
                base_class=nn.Module,
                builder=lambda cls: Sequential(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    layer_class=cls,
                    hidden_dims=hidden_dims,
                    act_fn=act_fn,
                    norm_fn=norm_fn,
                    end_fn=end_fn,
                    layer_kwargs=mlp_kwargs,
                )
            )

        # init pooling
        self.pooling = clone_or_init(
            name='pooling_class',
            obj=pooling_class,
            base_class=SetPooling,
            builder = lambda cls: cls(
                mask=torch.ones(self.n_dim,1),
                num_features=self.embed_dim,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn,
                **pooling_kwargs
            )
        )

    def _pool(self, x:Tensor, need_weights:bool):
        # get pooled embedding
        pool_out = self.pooling(x, concat=False, return_dict=need_weights)

        # extract output
        if isinstance(pool_out, Tensor):
            x = pool_out
        else:
            pool_out = input_to_dict(pool_out) # extract as dict
            x = pool_out.pop('x')

        return x, pool_out
        
    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # get input as dict
        data = input_to_dict(input)

        # mlp
        if self.mlp is not None:
            data['x'] = self.mlp(data['x'])

        # pool
        data['x'], lp_out = self._pool(data['x'], need_weights)
        data['x'] = data['x'].squeeze(1) # flatten to (b,F)

        if need_weights:
            data['layer_outs']['lp'] = lp_out
            data['z'] = data['x']

        return data

## Classifiers

@cloneable
class LatentClassifier(nn.Module):
    def __init__(
        self,
        dataset:GraphDataset, # dims
        embed_dim:int=None, # dims
        head_dim:int=None,  # dims
        num_heads:int=1,  # dims
        method:Literal['node','set']='node', # dims, encoder; twin removed for now

        # layers
        norm_class:Normalizer=None, # encoder
        encoder_class:nn.Module=None, # encoder
        pooling_class:SetPooling=SetPooling, # encoder, latent
        mlp:Union[bool,nn.Module]=False, # latent
        classifier:nn.Module=nn.Linear, # classifier

        # new layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        # kwargs
        norm_kwargs:dict=None, # encoder
        encoder_kwargs:dict=None, # encoder
        pooling_kwargs:dict=None, # encoder, latent
        mlp_kwargs:dict=None, # latent
        classifier_kwargs:dict=None, # classifier
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs
        pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        mlp_kwargs = {} if mlp_kwargs is None else mlp_kwargs
        classifier_kwargs = {} if classifier_kwargs is None else classifier_kwargs

        # get dims
        self.dims = Dims(
            dataset=dataset,
            embed_dim=embed_dim, head_dim=head_dim, num_heads=num_heads,
            method=method
        )

        # build hidden_dims from Dims
        hidden_dims = build_hidden_dims(self.dims.embed_dim, hidden_dims)

        # build model
        self.encoder = Encoder(
            dims=self.dims, method=method,
            norm_class=norm_class, encoder_class=encoder_class, pooling_class=pooling_class,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn,
            norm_kwargs=norm_kwargs, encoder_kwargs=encoder_kwargs, pooling_kwargs=pooling_kwargs
        )

        self.latent = Latent(
            dims=self.dims,
            mlp=mlp, pooling_class=pooling_class,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn,
            mlp_kwargs=mlp_kwargs, pooling_kwargs=pooling_kwargs
        )

        self.classifier = clone_or_init(
            name='classifier',
            obj=classifier,
            base_class=nn.Module,
            builder=lambda cls: Sequential(
                in_channels=self.dims.embed_dim,
                out_channels=self.dims.num_classes,
                layer_class=cls,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=False, # no final layer, output raw logits
                layer_kwargs=classifier_kwargs,
            )
        )

    def init_with_loader(self, loader): # pass loader to Encoder -> Norm
        if callable(getattr(self.encoder.norm, 'init_with_loader', None)):
            self.encoder.norm.init_with_loader(loader)

    def _get_logits(self, data:dict, need_weights:bool):
        # classify
        cls_out = self.classifier(data, return_dict=need_weights, return_attention_weights=need_weights)

        # extract logits from output
        if isinstance(cls_out, Tensor):
            logits = cls_out
        else:
            cls_out = input_to_dict(cls_out)
            logits = cls_out.pop('x')

        # can return cls_out if not nn.Linear; change in forward()
        return logits 

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # get latent embedding
        data = self.encoder(input, need_weights)
        data = self.latent(data, need_weights)
        
        # get logits
        logits = self._get_logits(data, need_weights)

        # format output
        del data['x']
        data['y_logits'] = logits
        data['y_probs'] = logits.softmax(dim=-1)
        data['y_preds'] = logits.argmax(dim=-1)

        return data

##