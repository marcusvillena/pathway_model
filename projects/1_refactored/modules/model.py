import torch
import torch.nn as nn

from modules.layers import SetPooling, Sequential
from modules.utils import attn_dims, cloneable, input_to_dict, reshape

from torch import Tensor
from torch_geometric.data import Data
from typing import Literal, Optional, Union

## Autoencoder
@cloneable
class Encoder(nn.Module):
    def __init__(
        self,
        # dims
        mask:Tensor=None,
        num_features:int=None,
        embed_dim:int=None,
        head_dim:int=None,
        num_heads:int=1,

        # layers; instance or predefined
        node_encoder:Union[nn.Module,Sequential]=None,
        set_pooling:SetPooling=None,

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        # etc
        method:Literal['node','set','twin']='node',
        x_mean:Optional[Tensor]=None,
        learn_mu:bool=False,
        eps:float=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # dims
        self.mask = mask
        self.num_features = num_features
        self.num_nodes, self.num_sets = mask.shape
        self.embed_dim, self.head_dim, self.num_heads = attn_dims(embed_dim, head_dim, num_heads)
        self.method = method
        self.eps = eps

        # nb
        self.log_theta = nn.Parameter(torch.zeros(1, self.num_nodes, 1))
        if isinstance(x_mean, Tensor):
            if learn_mu:
                self.log_mu = nn.Parameter(torch.log(x_mean+1).detach()) # learned with init
            else:
                self.register_buffer('log_mu', torch.log(x_mean+1).detach()) # fixed
        else:
            self.log_mu = nn.Parameter(torch.randn(1, self.num_nodes, 1) * 0.1 + 8.0) # exp() 8 +- 0.3
        
        # node encoder; init new, or copy if provided
        if isinstance(node_encoder, type) and issubclass(node_encoder, nn.Module):
            self.node_encoder = Sequential(
                in_channels=self.num_features,
                out_channels=self.embed_dim,
                layer_class=node_encoder,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )
        elif isinstance(node_encoder,(nn.Module, Sequential)):
            self.node_encoder = node_encoder.copy()
        else:
            raise TypeError(f'node_encoder must be a type, predefined nn.Module, or Sequential, got: {type(node_encoder)}')

        # set pooling
        if isinstance(set_pooling, type) and issubclass(set_pooling, SetPooling):
            self.set_pooling = set_pooling(
                mask=self.mask,
                num_features=self.embed_dim,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                norm_fn=norm_fn,
                end_fn=end_fn
            )
        elif isinstance(set_pooling, (nn.Module, SetPooling)):
            self.set_pooling = set_pooling.copy()
        else:
            self.set_pooling = None
    
    def forward(self, input:Union[Data, Tensor, dict], return_dict:bool=False, need_weights:bool=False, **kwargs):
        # extract x as b*n,f
        data = input_to_dict(input)
        x = reshape(data['x'], 'b*n,f', num_nodes=self.num_nodes, num_features=self.num_features)
        batch_size = x.shape[0] // self.num_nodes

        # global log nb
        log_mu = self.log_mu.expand(batch_size, self.num_nodes, 1).reshape(-1, 1) # use for lfc
        log_theta = self.log_theta.expand(batch_size, self.num_nodes, 1).reshape(-1, 1)
        mu = torch.exp(log_mu) # for nbloss
        theta = torch.exp(log_theta) # for nbloss

        # get ground truth (obs) lfc from global nb
        lfc = x - log_mu
        data['x'] = lfc # pass downstream/encoder

        # node embedding
        return_attention_weights=True if need_weights else None # PyG GATConv, GraphTransformer
        h = self.node_encoder(data, return_dict=need_weights, return_attention_weights=return_attention_weights, **kwargs)
        h = input_to_dict(h) # extract if dict
        x = reshape(h['x'], 'b,n,f', num_nodes=self.num_nodes, num_features=self.embed_dim)
        
        # pooling
        if self.method in ('set','twin'):
            assert self.set_pooling is not None, "set_pooling must be provided for method='set' or 'twin'."
            concat = False if self.method == 'set' else True
            pool_out = self.set_pooling(x, concat=concat, return_dict=need_weights)

            if need_weights:
                h.update(pool_out)
            else:
                x = pool_out

        # return
        if return_dict:
            h.update({'lfc':lfc,'mu':mu,'theta':theta})
            return h
        else:
            return x, lfc, mu, theta
        
    def get_weights(self, input:Union[Data, Tensor, dict], **kwargs):
        return self.forward(input=input, return_dict=True, need_weights=True)

@cloneable
class Latent(nn.Module):
    def __init__(
        self,
        # dims
        mask:Tensor=None,
        embed_dim:int=None,
        head_dim:int=None,
        num_heads:int=1,

        # mlp
        mlp:Union[bool,Sequential]=False,
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        # pooling
        global_pooling:SetPooling=None,
        method:Literal['node','set','twin']='node',
        fwd:Literal['node','set','twin','twin_pool']=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # dims
        self.mask = mask
        self.num_nodes, self.num_sets = mask.shape

        self.embed_dim, self.head_dim, self.num_heads = attn_dims(embed_dim, head_dim, num_heads)
        self.method = method
        
        # default fwd
        if fwd == None:
            fwd = method
        self.fwd = fwd

        # determine n dim (for reshaping)
        num_nodes, num_sets = mask.shape
        if method == 'node':
            self.n_dim = num_nodes
            self.split_dim = None
        elif method == 'set':
            self.n_dim = num_sets
            self.split_dim = None
        elif method == 'twin':
            self.n_dim = num_nodes + num_sets
            self.split_dim = [num_nodes, num_sets]
        else:
            raise TypeError(f'unsupported method: {method}')
        
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
        def init_pooling(pooling, mask, condition):
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
                    raise TypeError(f'global_pooling must be a type, predefined nn.Module, or SetPooling, got: {type(global_pooling)}')
            else:
                return None

        self.node_pool = init_pooling(global_pooling, torch.ones(self.num_nodes,1), method in ('node','twin'))
        self.set_pool = init_pooling(global_pooling, torch.ones(self.num_sets,1), method in ('set','twin'))
        self.twin_pool = init_pooling(global_pooling, torch.ones(2,1), fwd == 'twin_pool')

    def forward(self, input:Union[Data, Tensor, dict], return_dict:bool=True, need_weights:bool=False):
        # get input as kwargs dict
        data = input_to_dict(input)

        # get x in (batch, n, features), where n is (nodes) or (nodes+sets)
        x = reshape(x=data['x'], to='b,n,f', num_nodes=self.n_dim, num_features=self.embed_dim)

        # mlp
        x = self.mlp(x) if self.mlp is not None else x

        # pool
        if self.method == 'node':
            z_node = self.node_pool(x, concat=False)
            z_set = None
        elif self.method == 'set':
            z_node = None
            z_set = self.set_pool(x, concat=False)
        else: #self.method == 'twin'
            x_node, x_set = x.split(self.split_dim, dim=1) #split, pool sep
            z_node = self.node_pool(x_node, concat=False)
            z_set = self.set_pool(x_set, concat=False)

        # fwd
        if self.fwd == 'node':
            x = z_node
        elif self.fwd == 'set':
            x = z_set
        else: # self.fwd == 'twin' or 'twin_pool':
            x = torch.cat([z_node, z_set], dim=1)
            x = self.twin_pool(x, concat=False) if self.fwd == 'twin_pool' else x.mean(dim=1)

        # squeeze to (b,F)
        x = x.squeeze(1)
        z_node = z_node.squeeze(1) if z_node is not None else z_node
        z_set = z_set.squeeze(1) if z_set is not None else z_set

        # return
        return {'x':x, 'z_node':z_node, 'z_set':z_set} if return_dict else (x, z_node, z_set)

    def get_weights(self, input:Union[Data, Tensor, dict]):
        # extract x from input, pass through mlp if needed
        data = input_to_dict(input)
        x = reshape(x=data['x'], to='b,n,f', num_nodes=self.n_dim, num_features=self.embed_dim)
        x = self.mlp(x) if self.mlp is not None else x

        # pool w/ weights in dict
        if self.method == 'node':
            node_pool_out = self.node_pool(x, concat=False, return_dict=True)
            z_node = node_pool_out.get('x')
            attn_node = node_pool_out.get('attn')

            z_set = None
            attn_set = None

        elif self.method == 'set':
            set_pool_out = self.set_pool(x, concat=False, return_dict=True)
            z_set = set_pool_out.get('x')
            attn_set = set_pool_out.get('attn')

            z_node = None
            attn_node = None

        else: #self.method == 'twin'
            x_node, x_set = x.split(self.split_dim, dim=1) #split, pool sep

            node_pool_out = self.node_pool(x_node, concat=False, return_dict=True)
            z_node = node_pool_out.get('x')
            attn_node = node_pool_out.get('attn')

            set_pool_out = self.set_pool(x_set, concat=False, return_dict=True)
            z_set = set_pool_out.get('x')
            attn_set = set_pool_out.get('attn')

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
                twin_pool_out = self.twin_pool(x, concat=False, return_dict=True)
                x = twin_pool_out.get('x')
                attn_twin = twin_pool_out('attn')
            else:
                x = x.mean(dim=1)
                attn_twin = None

        return {
            'x': x,
            'z_node': z_node,
            'z_set': z_set,
            'attn_node': attn_node,
            'attn_set': attn_set,
            'attn_twin': attn_twin
        }

@cloneable
class Decoder(nn.Module):
    def __init__(
        self,
        # dims
        mask:Tensor=None,
        embed_dim:int=None,
        head_dim:int=None,
        num_heads:int=1,

        # layer args
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        # decoder
        expand_mlp:bool=False,
        shared_mlp:bool=True,
        fit_nb:bool=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # dims
        self.num_nodes, _ = mask.shape
        embed_dim, head_dim, num_heads = attn_dims(embed_dim, head_dim, num_heads)

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

    def forward(self, input:Union[Data, Tensor, dict], return_dict:bool=True):
        # get inputs as kwargs dict
        data = input_to_dict(input)
        z = data['x']
        batch_size = z.shape[0]

        # expand z (b,E) -> (b,n,E)
        z = self.expand(z)
        z = z.view(batch_size, self.num_nodes, -1)
        
        # estimate fc -> (b,n,1) -> reshape to original (b*n,1)
        lfc_recon = self.estimate(z).reshape(-1,1)
        
        if return_dict:
            return {'lfc_recon':lfc_recon}
        else:
            return lfc_recon

@cloneable
class Autoencoder(nn.Module):
    def __init__(
        self,
        # dims
        mask:Tensor=None,
        num_features:int=None, # encoder
        embed_dim:int=None,
        head_dim:int=None,
        num_heads:int=1,

        # layers
        node_encoder:Union[nn.Module,Sequential]=None, # encoder
        pooling:SetPooling=None, # encoder (set), latent (global)
        mlp:Union[bool,Sequential]=False, # latent

        # layer params
        hidden_dims:list[int]=None, 
        act_fn:nn.Module=None, 
        norm_fn:Literal['batch','layer']=None, 
        end_fn:Union[bool,nn.Module]=False,

        # etc
        method:Literal['node','set','twin']='node', # encoder, latent
        fwd:Literal['node','set','twin','twin_pool']=None, # latent
        expand_mlp:bool=False, # decoder
        shared_mlp:bool=True, # decoder
        fit_nb:bool=False, # decoder
        log_transform:bool=True, # autoencoder
        x_mean:Optional[Tensor]=None,
        learn_mu:bool=False,
    ):
        super().__init__()
        self.log_transform = log_transform

        # dims
        embed_dim, head_dim, num_heads = attn_dims(embed_dim, head_dim, num_heads)

        # modules
        self.encoder = Encoder(
            mask, num_features, embed_dim, head_dim, num_heads,
            node_encoder, pooling,
            hidden_dims, act_fn, norm_fn, end_fn,
            method, x_mean, learn_mu,
        )

        self.latent = Latent(
            mask, embed_dim, head_dim, num_heads,
            mlp,
            hidden_dims, act_fn, norm_fn, end_fn,
            pooling, method, fwd
        )

        self.decoder = Decoder(
            mask, embed_dim, head_dim, num_heads,
            hidden_dims, act_fn, norm_fn, end_fn,
            expand_mlp, shared_mlp, fit_nb
        )

    def forward(self, input:Union[Data, Tensor, dict], return_dict:bool=True):
        # get input as kwargs dict
        data = input_to_dict(input)

        if self.log_transform:
            data['x'] = torch.log(data['x'] + 1)

        # return tensor h (h_n, h_s, or h_ns)
        h, lfc, mu, theta = self.encoder(data, return_dict=False) # return_attention_weights=None

        # get z
        z, _, _ = self.latent(h, return_dict=False)

        # get fc_recon
        lfc_recon = self.decoder(z, return_dict=False)

        # get x_recon
        x_recon = torch.exp(lfc_recon) * mu
        # x_recon = mu

        if return_dict:
            return {'x_recon':x_recon, 'lfc_recon':lfc_recon, 'lfc':lfc, 'mu':mu, 'theta':theta}
        else:
            return x_recon, lfc_recon, lfc, mu, theta
        
    def get_weights(self, input:Union[Data, Tensor, dict]):
        # get input as kwargs dict
        data = input_to_dict(input)

        if self.log_transform:
            data['x'] = torch.log(data['x'] + 1)

        # encoder
        encoder_out = self.encoder.get_weights(data)
        h = encoder_out.get('x')
        attn_n2s = encoder_out.get('attn')
        lfc = encoder_out.get('lfc')
        mu = encoder_out.get('mu')
        theta = encoder_out.get('theta')

        # latent
        latent_out = self.latent.get_weights(h)
        z = latent_out.get('x')
        z_node = latent_out.get('z_node')
        z_set = latent_out.get('z_set')
        attn_n2z = latent_out.get('attn_node')
        attn_s2z = latent_out.get('attn_set')
        attn_twin = latent_out.get('attn_twin')

        # decoder, recon
        lfc_recon = self.decoder(z, return_dict=False)
        x_recon = torch.exp(lfc_recon) * mu

        return {
            'x_recon': x_recon,
            'lfc_recon': lfc_recon,
            'lfc': lfc,
            'mu': mu,
            'theta': theta,
            'h': h,
            'z': z,
            'z_node': z_node,
            'z_set': z_set,
            'attn_n2s': attn_n2s,
            'attn_n2z': attn_n2z,
            'attn_s2z': attn_s2z,
            'attn_twin': attn_twin
        }
