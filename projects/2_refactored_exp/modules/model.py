
from .layers import Dims, SetPooling, Sequential
from .utils import build_hidden_dims, cloneable, clone_or_init, input_to_dict
import torch
import torch.nn as nn

# typing
from .data import GraphDataset
from .norm import Normalizer
from torch import Tensor
from torch_geometric.data import Data
from typing import Literal, Union, overload

## General
    
@cloneable
class Encoder(nn.Module):
    def __init__(
        self,
        dims: Dims,
        method: Literal['node','set'] = 'node', # twin removed for now

        # layers
        norm_class: Normalizer | type[Normalizer] = Normalizer,
        encoder_class: nn.Module | type[nn.Module] = nn.Linear,
        pooling_class: SetPooling | type[SetPooling] | None = None,

        # new layer params
        hidden_dims: int | list[int] | None = None, 
        act_fn: nn.Module | None = None, 
        norm_fn: Literal['batch','layer'] | None = None, 
        end_fn: bool | nn.Module = False,

        # kwargs
        norm_kwargs: dict | None = None,
        encoder_kwargs: dict | None = None,
        pooling_kwargs: dict | None = None,

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
        self.norm = clone_or_init(
            name='norm_class',
            obj=norm_class,
            base_class=nn.Module,
            builder=lambda cls: cls(**norm_kwargs)
        )

        # init layers
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
        elif (method == 'set') and (pooling_class is not None):
            self.node_pooling = clone_or_init(
                name='pooling_class',
                obj=pooling_class,
                base_class=SetPooling,
                builder = lambda cls: cls(
                    dims=dims,
                    mask=self.mask,
                    num_features=self.embed_dim,
                    hidden_dims=hidden_dims,
                    act_fn=act_fn,
                    norm_fn=norm_fn,
                    end_fn=end_fn,
                    **pooling_kwargs
                )
            )
        else: # method != node, and pooling_class == None
            raise ValueError(f"'pooling_class' must be a SetPooling type or instance. Got: {type(pooling_class)}")


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
        h_node = h_node.reshape(-1, self.num_nodes, self.embed_dim)

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
        self.orig_shape = data['x'].shape

        # normalize
        data = self._normalize(data)
        x_t = data['x'] # x_true (model space) for reconstr

        # node embedding
        x, ne_out = self._encode(data, need_weights, **kwargs)
        h_node = x
        
        # node pooling
        x, np_out = self._pool(x, need_weights)
        h_pool = x if self.node_pooling is not None else None

        # format output
        out = {}
        out['x'] = x
        out['x_t'] = x_t
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
        dims: Dims,

        # layers
        mlp: bool | nn.Module = False,
        pooling_class: SetPooling | type[SetPooling] = SetPooling,

        # new layer params
        hidden_dims: int | list[int] | None = None, 
        act_fn: nn.Module | None = None, 
        norm_fn: Literal['batch','layer'] | None = None, 
        end_fn: bool | nn.Module = False,

        # kwargs
        mlp_kwargs: dict | None = None,
        pooling_kwargs: dict | None = None,
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
                dims=dims,
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

@cloneable
class BaseModel(nn.Module):
    def __init__(
        self,
        dataset: GraphDataset, # dims
        out_dim: int, # output
        embed_dim: int | None = None, # dims
        head_dim: int | None = None,  # dims
        num_heads: int = 1,  # dims
        method: Literal['node','set'] = 'node', # dims, encoder; twin removed for now

        # layers
        norm_class: Normalizer | None = None, # encoder
        encoder_class: nn.Module | type[nn.Module] | None = None, # encoder
        pooling_class: SetPooling | type[SetPooling] = SetPooling, # encoder, latent
        mlp: bool | nn.Module = False, # latent
        out_module: nn.Module | type[nn.Module] = nn.Linear, # output

        # new layer params
        hidden_dims: int | list[int] | None = None, 
        act_fn: nn.Module | None = None, 
        norm_fn: Literal['batch','layer'] | None = None, 
        end_fn: bool | nn.Module = False,

        # kwargs
        norm_kwargs: dict | None = None, # encoder
        encoder_kwargs: dict | None = None, # encoder
        pooling_kwargs: dict | None = None, # encoder, latent
        mlp_kwargs: dict | None = None, # latent
        out_kwargs: dict | None = None, # output
    ):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs
        pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        mlp_kwargs = {} if mlp_kwargs is None else mlp_kwargs
        out_kwargs = {} if out_kwargs is None else out_kwargs

        # get dims
        self.dims = Dims(
            dataset=dataset,
            embed_dim=embed_dim, head_dim=head_dim, num_heads=num_heads,
            method=method
        )

        # build hidden_dims from Dims
        self.hidden_dims = build_hidden_dims(self.dims.embed_dim, hidden_dims)

        # build model
        self.encoder = Encoder(
            dims=self.dims, method=method,
            norm_class=norm_class, encoder_class=encoder_class, pooling_class=pooling_class,
            hidden_dims=self.hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn,
            norm_kwargs=norm_kwargs, encoder_kwargs=encoder_kwargs, pooling_kwargs=pooling_kwargs
        )

        self.latent = Latent(
            dims=self.dims,
            mlp=mlp, pooling_class=pooling_class,
            hidden_dims=self.hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn,
            mlp_kwargs=mlp_kwargs, pooling_kwargs=pooling_kwargs
        )

        self.out_layer = clone_or_init(
            name='out_layer',
            obj=out_module,
            base_class=nn.Module,
            builder=lambda out_class: Sequential(
                in_channels=self.dims.embed_dim,
                out_channels=out_dim,
                layer_class=out_class,
                hidden_dims=self.hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=False, # no final layer, output raw logits
                layer_kwargs=out_kwargs,
            )
        )

    def init_with_loader(self, loader): # pass loader to Encoder -> Norm
        if callable(getattr(self.encoder.norm, 'init_with_loader', None)):
            self.encoder.norm.init_with_loader(loader)

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # get latent embedding
        data = self.encoder(input, need_weights)
        data = self.latent(data, need_weights)

        # get output
        out = self.out_layer(data, return_dict=need_weights, return_attention_weights=need_weights)

        # extract x from output
        if isinstance(out, Tensor):
            x = out
            out = None
        else:
            out = input_to_dict(out)
            x = out.pop('x')

        # pass to child class via super.forward()
        return x, out, data

## Models

@cloneable
class BaseClassifier(BaseModel):
    def __init__(
        self,
        dataset: GraphDataset, # dims
        out_dim: int | None = None, # output
        embed_dim: int | None = None, # dims
        head_dim: int | None = None,  # dims
        num_heads: int = 1,  # dims
        method: Literal['node','set'] = 'node', # dims, encoder; twin removed for now

        # layers
        norm_class: Normalizer | None = None, # encoder
        encoder_class: nn.Module | type[nn.Module] | None = None, # encoder
        pooling_class: SetPooling | type[SetPooling] = SetPooling, # encoder, latent
        mlp: bool | nn.Module = False, # latent
        out_module: nn.Module | type[nn.Module] = nn.Linear, # output

        # new layer params
        hidden_dims: int | list[int] | None = None, 
        act_fn: nn.Module | None = None, 
        norm_fn: Literal['batch','layer'] | None = None, 
        end_fn: bool | nn.Module = False,

        # kwargs
        norm_kwargs: dict | None = None, # encoder
        encoder_kwargs: dict | None = None, # encoder
        pooling_kwargs: dict | None = None, # encoder, latent
        mlp_kwargs: dict | None = None, # latent
        out_kwargs: dict | None = None, # output
    ):
        # default: out_dim = num_classes (for classification)
        out_dim = dataset.num_classes if out_dim is None else out_dim

        # call init
        super().__init__(
            dataset, out_dim, embed_dim, head_dim, num_heads, method,
            norm_class, encoder_class, pooling_class, mlp, out_module,
            hidden_dims, act_fn, norm_fn, end_fn,
            norm_kwargs, encoder_kwargs, pooling_kwargs, mlp_kwargs, out_kwargs
        )

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        x, out, data = super().forward(input, need_weights)
        
        # format output
        del data['x'] # avoid overlap with batch dict
        data['y_logits'] = x
        data['y_probs'] = x.softmax(dim=-1)
        data['y_preds'] = x.argmax(dim=-1)
        if need_weights:
            data['layer_outs']['clf'] = out
        return data

@cloneable
class BaseAutoencoder(BaseModel):
    def __init__(
        self,
        dataset: GraphDataset, # dims
        out_dim: int | None = None, # output
        embed_dim: int | None = None, # dims
        head_dim: int | None = None,  # dims
        num_heads: int = 1,  # dims
        method: Literal['node','set'] = 'node', # dims, encoder; twin removed for now

        # layers
        norm_class: Normalizer | None = None, # encoder
        encoder_class: nn.Module | type[nn.Module] | None = None, # encoder
        pooling_class: SetPooling | type[SetPooling] = SetPooling, # encoder, latent
        mlp: bool | nn.Module = False, # latent
        out_module: nn.Module | type[nn.Module] = nn.Linear, # output

        # new layer params
        hidden_dims: int | list[int] | None = None, 
        act_fn: nn.Module | None = None, 
        norm_fn: Literal['batch','layer'] | None = None, 
        end_fn: bool | nn.Module = False,

        # kwargs
        norm_kwargs: dict | None = None, # encoder
        encoder_kwargs: dict | None = None, # encoder
        pooling_kwargs: dict | None = None, # encoder, latent
        mlp_kwargs: dict | None = None, # latent
        out_kwargs: dict | None = None, # output
    ):
        # default: out_dim = num_classes (for classification)
        out_dim = dataset[0].num_nodes if out_dim is None else out_dim

        # call init
        super().__init__(
            dataset, out_dim, embed_dim, head_dim, num_heads, method,
            norm_class, encoder_class, pooling_class, mlp, out_module,
            hidden_dims, act_fn, norm_fn, end_fn,
            norm_kwargs, encoder_kwargs, pooling_kwargs, mlp_kwargs, out_kwargs
        )

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        x, out, data = super().forward(input, need_weights)
        
        # format output
        del data['x'] # avoid overlap with batch dict
        data['x_preds'] = x.view(self.encoder.orig_shape)
        if need_weights:
            data['layer_outs']['dec'] = out
        return data