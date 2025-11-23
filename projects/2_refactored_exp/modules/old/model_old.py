from ..data import Preprocessor
from ..layers import SetPooling, Sequential
from ..utils import attn_dims, cloneable, input_to_dict, reshape

import torch
import torch.nn as nn

from torch import Tensor
from torch_geometric.data import Data
from typing import Literal, Optional, Union

# general
@cloneable
class Dims():
    def __init__(
        self, 
        data:Preprocessor, 
        embed_dim:int=None, 
        head_dim:int=None, 
        num_heads:int=1, 
        method:Literal['node','set','twin']='node', 
        eps:float=1e-6
    ):
        self._data = data
        self.mask = data.pathway_index # legacy / compatibility
        self.num_sets = data.num_pathways # legacy / compatibility
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
        return getattr(self._data, attr)
    
    def __dir__(self):
        # list _data attr in IDEs
        return list(self.__dict__.keys()) + dir(self._data)
    
@cloneable
class NBParam(nn.Module):
    def __init__(self, dims:Dims, x_mean:Optional[Tensor]=None, learn_mu:bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = dims.num_nodes
        self.num_node_features = dims.num_node_features
        self.learn_mu = learn_mu

        # log_theta: learned, init at zero, shape (1, nodes, 1)
        self.log_theta = nn.Parameter(torch.zeros(1, self.num_nodes, 1))

        # log_mu: based on x_mean
        self._set_x_mean(x_mean)

    def init_with_loader(self, loader):
        self._set_x_mean(loader.mean)

    def _set_x_mean(self, x_mean):
        # generate log_mu if not provided; shape (1, nodes, 1)
        if isinstance(x_mean, Tensor):
            x_mean = torch.log1p(x_mean).detach()
        else:
            x_mean = torch.randn(1, self.num_nodes, 1) * 0.1 + 8.0 # empirical; exp() 8 +- 0.3

        # set log_mu as learnable param or not
        if self.learn_mu:
            self.log_mu = nn.Parameter(x_mean)
        else:
            self.register_buffer('log_mu', x_mean)

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # extract x as b*n,f
        data = input_to_dict(input)
        x = reshape(data['x'], 'b*n,f', num_nodes=self.num_nodes, num_features=self.num_node_features)
        batch_size = x.shape[0] // self.num_nodes

        # expand log_mu, log_theta to batch size
        log_mu = self.log_mu.expand(batch_size, self.num_nodes, 1).reshape(-1, 1) # use for lfc
        log_theta = self.log_theta.expand(batch_size, self.num_nodes, 1).reshape(-1, 1)

        # convert to mu, theta (for nbloss)
        mu = torch.exp(log_mu)
        theta = torch.exp(log_theta)

        # get lfc
        lfc = torch.log1p(x) - log_mu

        # return as dict
        data['x'] = lfc # x is now lfc; pass to node encoder
        data['lfc'] = lfc # for analysis
        data['mu'] = mu # for nbloss, analysis
        data['theta'] = theta # for nbloss, analysis            

        return data

# encoder    
@cloneable
class NodeEncoder(nn.Module):
    def __init__(
        self,
        dims:Dims,
        layer_class:Union[nn.Module,Sequential]=None,
        
        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,
        
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_nodes = dims.num_nodes
        self.num_node_features = dims.num_node_features
        self.embed_dim = dims.embed_dim

        # init new
        if isinstance(layer_class, type) and issubclass(layer_class, nn.Module):
            self.node_encoder = Sequential(
                in_channels=self.num_node_features,
                out_channels=self.embed_dim,
                layer_class=layer_class,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )

        # copy if pre-init model is provided
        elif isinstance(layer_class,(nn.Module, Sequential)):
            self.node_encoder = layer_class.copy()

        # err
        else:
            raise TypeError(f'layer_class must be a type, predefined nn.Module, or Sequential, got: {type(layer_class)}')

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False, **kwargs):
        # extract x (or lfc)
        data = input_to_dict(input)
        data['x'] = data['x'].float()

        # return attention weights (PyG GATConv, GraphTransformer)
        return_attention_weights=True if need_weights else False

        # get node embedding
        ne_out = self.node_encoder(data, return_dict=need_weights, return_attention_weights=return_attention_weights, **kwargs)
        
        # extract node embedding from output
        if isinstance(ne_out, Tensor):
            h = ne_out
        else:
            ne_out = input_to_dict(ne_out) # extract as dict
            h = ne_out['x']

        # construct new dict (avoid copying graph data)
        out = {}
        out['x'] = h # x is now h_node, pass to n2s pooling
        # out['x_target'] = data['x']
        out['lfc'] = data.get('lfc')
        out['mu'] = data.get('mu')
        out['theta'] = data.get('theta')

        # for analysis:
        if need_weights:
            out['h_ne'] = h
            out['ne_out'] = ne_out

        return out

@cloneable
class NodePooling(nn.Module):
    def __init__(
        self,
        dims:Dims,
        pooling_class:SetPooling,
        method:Literal['set','twin']='set',

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask = dims.mask
        self.num_nodes = dims.num_nodes
        self.embed_dim = dims.embed_dim
        self.method = method

        # init new
        if isinstance(pooling_class, type) and issubclass(pooling_class, SetPooling): 
            self.set_pooling = pooling_class(
                mask=self.mask,
                num_features=self.embed_dim,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )
        # copy if pre-init class provided
        elif isinstance(pooling_class, (nn.Module, SetPooling)): 
            self.set_pooling = pooling_class.copy()
        else:
            raise TypeError(f'pooling_class must be a type, predefined')

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # extract h_node
        data = input_to_dict(input)
        h_node = reshape(data['x'], 'b,n,f', num_nodes=self.num_nodes, num_features=self.embed_dim)

        # pool node emb to set emb; dict if need_weights, else tensor
        concat = True if self.method == 'twin' else False
        pool_out = self.set_pooling(h_node, concat=concat, return_dict=need_weights)

        # return as dict
        if need_weights: # pool_out = dict(x, attn)
            data['x'] = pool_out['x']
            data['h_pool'] = pool_out['x']
            data['attn_n2s'] = pool_out['attn']
        else: # pool_out = Tensor(x)
            data['x'] = pool_out
        
        return data
        
@cloneable
class Encoder(nn.Module):
    def __init__(
        self,
        dims:Dims,

        # nb
        nb:bool=False,
        x_mean:Optional[Tensor]=None,
        learn_mu:bool=True,

        # encoder, pooling
        encoder_class:Union[nn.Module,Sequential]=None,
        pooling_class:Optional[SetPooling]=None,
        method:Literal['node','set','twin']='set',

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_nodes = dims.num_nodes
        self.num_node_features = dims.num_node_features

        # nb (optional)
        if nb: # init new
            self.nb = NBParam(
                dims=dims,
                x_mean=x_mean,
                learn_mu=learn_mu
            )
        else:
            self.nb = None

        # encoder (required)
        self.encoder = NodeEncoder(
            dims=dims,
            layer_class=encoder_class,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn
        )
        
        # pooling (optional)
        if (pooling_class is not None) and (method != 'node'):
            self.pooling = NodePooling(
                dims=dims,
                pooling_class=pooling_class,
                method=method,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )
        else:
            self.pooling = None

    def init_with_loader(self, loader): # pass loader to NBParam
        if hasattr(self.nb, 'init_with_loader'):
            self.nb.init_with_loader(loader)

    def get_log1p(self, input:Union[Data, Tensor, dict]):
        # extract x as b*n,f
        data = input_to_dict(input)
        x = reshape(data['x'], 'b*n,f', num_nodes=self.num_nodes, num_features=self.num_node_features)

        # get log1p
        x = torch.log1p(x)

        # return as dict
        data['x'] = x # x is now lfc; pass to node encoder       

        return data

    def forward(self, x:Union[Data, Tensor, dict], need_weights:bool=False):
        # nb pass
        if self.nb is not None:
            x = self.nb(x, need_weights)
        else:
            x = self.get_log1p(x)

        # encoder pass
        x = self.encoder(x, need_weights)

        # pooling pass
        if self.pooling is not None:
            x = self.pooling(x, need_weights)

        return x

@cloneable
class Latent(nn.Module):
    def __init__(
        self,
        dims:Dims,
        pooling_class:SetPooling,
        mlp:Union[bool,Sequential]=False,
        
        # method
        method:Literal['node','set','twin']='node',
        fwd:Literal['node','set','twin','twin_pool']=None,

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,        

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # dims
        self.embed_dim = dims.embed_dim
        self.num_nodes = dims.num_nodes
        self.num_sets = dims.num_sets
        self.n_dim = dims.n_dim
        self.split_dim = dims.split_dim

        # method
        self.method = method
        self.fwd = method if fwd is None else fwd
        
        # mlp
        if mlp:
            self.mlp = Sequential(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                layer_class=nn.Linear,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )
        elif isinstance(mlp, (nn.Module, Sequential)):
            self.mlp = mlp.clone()
        else:
            self.mlp = None

        # pooling
        def _init_pooling(pooling, mask, condition): # helper fxn
            if condition:
                if isinstance(pooling, type) and issubclass(pooling, SetPooling):
                    return pooling(
                        mask=mask,
                        num_features=self.embed_dim,
                        hidden_dims=hidden_dims,
                        act_fn=act_fn,
                        norm_fn=norm_fn,
                        end_fn=end_fn
                    )
                elif isinstance(pooling, (nn.Module, SetPooling)):
                    return pooling.copy(mask=mask)
                else:
                    raise TypeError(f'pooling_class must be a type, predefined nn.Module, or SetPooling, got: {type(pooling_class)}')
            else:
                return None

        self.node_pool = _init_pooling(pooling_class, torch.ones(self.num_nodes,1), method in ('node','twin'))
        self.set_pool = _init_pooling(pooling_class, torch.ones(self.num_sets,1), method in ('set','twin'))
        self.twin_pool = _init_pooling(pooling_class, torch.ones(2,1), fwd == 'twin_pool')

    def _pooling(self, pooling_class, x, need_weights:bool=False):
        if need_weights: # pooling outputs dict
            out = pooling_class(x, concat=False, return_dict=need_weights)
            z = out.get('x')
            attn = out.get('attn')

        else: # pooling outputs z (Tensor)
            z = pooling_class(x, concat=False, return_dict=need_weights)
            attn = None

        return z, attn

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # get input as kwargs dict
        data = input_to_dict(input)

        # get x in (batch, n, features), where n is (nodes) or (nodes+sets)
        x = reshape(x=data['x'], to='b,n,f', num_nodes=self.n_dim, num_features=self.embed_dim)

        # mlp
        if self.mlp is not None:
            x = self.mlp(x)
        if need_weights:
            data['h'] = x # h will be mlp output > h_pool > h_ne

        # pool
        if self.method == 'node':
            z_node, attn_node = self._pooling(self.node_pool, x, need_weights)
            z_set, attn_set = None, None
        elif self.method == 'set':
            z_node, attn_node = None, None
            z_set, attn_set = self._pooling(self.set_pool, x, need_weights)
        else: # self.method == 'twin'%%!
            x_node, x_set = x.split(self.split_dim, dim=1) #split, pool sep
            z_node, attn_node = self._pooling(self.node_pool, x_node, need_weights)
            z_set, attn_set = self._pooling(self.set_pool, x_set, need_weights)

        # fwd
        if self.fwd == 'node':
            x = z_node
            attn_twin = None
        elif self.fwd == 'set':
            x = z_set
            attn_twin = None
        else: # self.fwd == 'twin' or 'twin_pool':
            x = torch.cat([z_node, z_set], dim=1)
            if self.fwd == 'twin_pool':
                x, attn_twin = self._pooling(self.twin_pool, x, need_weights)
            else:
                x = x.mean(dim=1)
                attn_twin = None

        # squeeze to (b,F)
        x = x.squeeze(1)
        z_node = z_node.squeeze(1) if z_node is not None else z_node
        z_set = z_set.squeeze(1) if z_set is not None else z_set

        # return
        data['x'] = x
        if need_weights:
            data.update({
            'z': x,
            'z_node': z_node,
            'z_set': z_set,
            'attn_n2z': attn_node,
            'attn_s2z': attn_set,
            'attn_twin': attn_twin
        })
            
        return data

# autoencoder
@cloneable
class Decoder(nn.Module):
    def __init__(
        self,
        dims:Dims,
        nb:bool=False,

        # layer args
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # dims
        self.num_nodes = dims.num_nodes
        embed_dim = dims.embed_dim
        

        # node + sample estimate
        self.expand = Sequential(
            in_channels=embed_dim,
            out_channels=self.num_nodes * embed_dim,
            layer_class=nn.Linear,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn
        )

        self.estimate = Sequential(
            in_channels=embed_dim,
            out_channels=1,
            layer_class=nn.Linear,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn
        )

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # get inputs as kwargs dict
        data = input_to_dict(input)
        z = data['x']
        batch_size = z.shape[0]

        # expand z (b,E) -> (b,n,E)
        z = self.expand(z)
        z = z.view(batch_size, self.num_nodes, -1)
        
        # estimate fc -> (b,n,1) -> reshape to original (b*n,1)
        # this is a reconstruction of node encoder input, x_target (x, log_x, or lfc)
        # (x_target, x_pred) loss can suppl. NBLoss (e.g. if log_x or lfc)
        dec_out = self.estimate(z).reshape(-1,1)

        data['x'] = dec_out
        return data

@cloneable
class Autoencoder(nn.Module):
    def __init__(
        self,
        # dims
        data:Preprocessor, 
        embed_dim:int=None, 
        head_dim:int=None, 
        num_heads:int=1, 

        # nb
        nb:bool=False, # encoder (nb): decides to use nb (lfc)
        x_mean:Optional[Tensor]=None, # encoder (nb): baseline mean for nb
        learn_mu:bool=True, # encoder (nb): set mu as learnable param

        # encoder, pooling, mlp layers
        encoder_class:Union[nn.Module,Sequential]=None, # encoder: GNN class
        pooling_class:Optional[SetPooling]=None, # encoder, latent: pooling layer class
        mlp:Union[bool,Sequential]=False, # latent: use MLP after pooling

        # method
        method:Literal['set','twin']='set', # encoder, latent
        fwd:Literal['node','set','twin','twin_pool']=None, # latent

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nb = nb
        
        self.dims = Dims(
            data=data,
            embed_dim=embed_dim, head_dim=head_dim, num_heads=num_heads,
            method=method
        )

        self.encoder = Encoder(
            dims=self.dims, 
            nb=nb, x_mean=x_mean, learn_mu=learn_mu,
            encoder_class=encoder_class, pooling_class=pooling_class,
            method=method,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn
        )

        self.latent = Latent(
            dims=self.dims,
            pooling_class=pooling_class, mlp=mlp,
            method=method, fwd=fwd,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn
        )

        self.decoder = Decoder(
            dims=self.dims,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn
        )

    def init_with_loader(self, loader): # pass loader to Encoder -> NBParam
        if hasattr(self.encoder.nb, 'init_with_loader'):
            self.encoder.nb.init_with_loader(loader)

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        x = self.encoder(input, need_weights)
        x = self.latent(x, need_weights)
        x = self.decoder(x, need_weights)

        if self.nb:
            x['lfc_recon'] = x['x'] # decoder output is lfc_recon
            x['x'] = torch.exp(x['x']) * x['mu'] # convert to x_recon
        else:
            x['lfc_recon'] = None

        x['x_recon'] = x.pop('x') # rename x to x_recon to avoid confusion

        return x
    
    def get_weights(self, input:Union[Data, Tensor, dict]):
        x = self.forward(input, need_weights=True)
        return x
    
# classifier
@cloneable
class ClassifierLayer(nn.Module):
    def __init__(
        self,
        dims:Dims,

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,  

        *args, **kwargs  
    ):
        super().__init__(*args, **kwargs)
        self.embed_dim = dims.embed_dim
        self.num_classes = dims.num_classes
        self.n_dim = dims.n_dim

        self.mlp = Sequential(
            in_channels=self.embed_dim,
            out_channels=self.num_classes,
            layer_class=nn.Linear,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            norm_fn=norm_fn,
            end_fn=end_fn
        )

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        # input z (batch_size, embed_size)
        data = input_to_dict(input)
        x = data['x']
        y = {}
        
        # get logits
        x = self.mlp(x)
        y['y_logits'] = x # (batch_size, num_classes)

        # get probs, preds
        # if need_weights:
        y['y_probs'] = torch.softmax(x, dim=-1) # (batch_size,)
        y['y_preds'] = torch.argmax(x, dim=-1) # (batch_size,)

        return y

@cloneable
class Classifier(nn.Module):
    def __init__(
        self,
        # dims
        data:Preprocessor, 
        embed_dim:int=None, 
        head_dim:int=None, 
        num_heads:int=1, 

        # nb
        nb:bool=False, # encoder (nb): decides to use nb (lfc)
        x_mean:Optional[Tensor]=None, # encoder (nb): baseline mean for nb
        learn_mu:bool=True, # encoder (nb): set mu as learnable param

        # encoder, pooling, mlp layers
        encoder_class:Union[nn.Module,Sequential]=None, # encoder: GNN class
        pooling_class:Optional[SetPooling]=None, # encoder, latent: pooling layer class
        mlp:Union[bool,Sequential]=False, # latent: use MLP after pooling

        # method
        method:Literal['set','twin']='set', # encoder, latent
        fwd:Literal['node','set','twin','twin_pool']=None, # latent

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nb = nb
        
        self.dims = Dims(
            data=data,
            embed_dim=embed_dim, head_dim=head_dim, num_heads=num_heads,
            method=method
        )

        self.encoder = Encoder(
            dims=self.dims, 
            nb=nb, x_mean=x_mean, learn_mu=learn_mu,
            encoder_class=encoder_class, pooling_class=pooling_class,
            method=method,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn
        )

        self.latent = Latent(
            dims=self.dims,
            pooling_class=pooling_class, mlp=mlp,
            method=method, fwd=fwd,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn
        )

        self.classifier = ClassifierLayer(
            dims=self.dims,
            hidden_dims=hidden_dims, act_fn=act_fn, norm_fn=norm_fn, end_fn=end_fn
        )

    def init_with_loader(self, loader): # pass loader to Encoder -> NBParam
        if hasattr(self.encoder.nb, 'init_with_loader'):
            self.encoder.nb.init_with_loader(loader)

    def forward(self, input:Union[Data, Tensor, dict], need_weights:bool=False):
        x = self.encoder(input, need_weights)
        x = self.latent(x, need_weights)
        y = self.classifier(x, need_weights)

        x.update(y)

        return x
    
    def get_weights(self, input:Union[Data, Tensor, dict]):
        x = self.forward(input, need_weights=True)
        return x
