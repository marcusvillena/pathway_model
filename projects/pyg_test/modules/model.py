import torch
import torch.nn as nn
import torch.nn.functional as F

# type hinting
from pandas import DataFrame
from torch import Tensor
from typing import Literal, Callable

#### functions ####

def _get_edge_index(relation:DataFrame, method:Literal['out','in']='out', source:str='idx1', target:str='idx2'):
    # get edge information >> (num_edges, 2)
    if method == 'out':
        edge_index = relation[[source, target]]
    elif method == 'in':
        edge_index = relation[[target, source]]
    
    # transpose, convert to tensor >> (2, num_edges) 
    edge_index = Tensor(edge_index.values.T).to(dtype=torch.int64)

    return edge_index

def _get_adj(relation:DataFrame, num_nodes:int, method:Literal['out','in']='out', source:str='idx1', target:str='idx2') -> Tensor:
    # get edge information
    edge_indices = Tensor(relation[[source, target]].values).int() # (num_edges, 2)

    # init adj
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32) # (num_nodes, num_nodes)

    # compute adj
    if method == 'in':
        adj[edge_indices[:,1], edge_indices[:,0]] = 1 # in_adj if specified
    else:
        adj[edge_indices[:,0], edge_indices[:,1]] = 1 # out_adj otherwise (default)

    return adj

def _norm_adj(adj:Tensor, method:Literal['sym','row']=None) -> Tensor:
    if method != None:
        # get degree (row sum)
        degree = adj.sum(dim=-1)

    # row normalization
    if method == 'row':
        # get 1/degree
        degree_inv = torch.where(degree != 0, 1/degree, torch.zeros_like(degree))

        # get D (diagonalized degree matrix)
        D_inv = torch.diag(degree_inv)

        # normalize (matmul)
        return D_inv @ adj
    
    # symmetric normalization
    elif method == 'sym':
        # get 1/sqrt(degree)
        degree_inv_sqrt = torch.where(degree != 0, 1/torch.sqrt(degree), torch.zeros_like(degree))

        # get D (diagonalized degree matrix)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        # normalize (matmul)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    else:
        return adj
    
def _get_layers(in_features:int, out_features:int, layer_class, layer_kwargs:dict, hidden_dims:list=[], act_fn=nn.LeakyReLU(), end_fn=None):
    # init layers
    num_layers = len(hidden_dims) # set num_layers = len of hidden_dims
    layers = [] # init empty layers list
    in_dim = in_features # first layer input = in_features (num_features)

    # hidden layer append loop
    for i in range(num_layers):

        if i < len(hidden_dims): # set out_dim = hidden dim if applicable
            out_dim = hidden_dims[i]

        else: # else out_dim = (final) output dim
            out_dim = out_features

        # append layer and activation function
        layers.append(layer_class(in_features=in_dim, out_features=out_dim, **layer_kwargs))
        layers.append(act_fn)

        # next layer in_dim = current out_dim
        in_dim = out_dim

    # append final layer
    layers.append(layer_class(in_features=in_dim, out_features=out_features, **layer_kwargs))

    # append end fn, if applicable
    if end_fn != None:
        layers.append(end_fn)

    return layers

#### MLP ####

class MLP(nn.Module):
    def __init__(self, in_features:int, out_features:int, hidden_dims:list=[], bias:bool=False, act_fn=nn.LeakyReLU(), end_fn=None):
        super().__init__()

        # define layers
        layers = _get_layers(
            in_features=in_features,
            out_features=out_features,
            layer_class=nn.Linear,
            layer_kwargs={'bias':bias},
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            end_fn=end_fn
        )

        # define model
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class MLPClassifier(nn.Module):
    def __init__(self, in_features:int, out_features:int, mlp_kwargs:dict={}, flatten:bool=False):
        super().__init__()

        # assign instance variables
        self.flatten = flatten

        # define layers
        self.mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            **mlp_kwargs
        )

    def forward(self, X:Tensor):
        # flatten if applicable
        if self.flatten == True:
            X = X.squeeze(-1)

        # forward pass
        logits = self.mlp(X)

        return logits

    def predict(self, X:Tensor, as_logits:bool=True):
        # transform if raw data (not logits)
        if as_logits == False:
            X = self.forward(X)

        # convert logits to prediction
        probs = torch.softmax(X, dim=1) # softmax to probs
        y_pred = torch.argmax(probs, dim=1) # argmax to most likely class
        y_pred = F.one_hot(y_pred, probs.shape[1]) # get one-hot encoding

        return y_pred
    
class MLPAutoencoder(nn.Module):
    def __init__(self, in_features:int, embedding_size:int=8, encoder_kwargs:dict={}, decoder_kwargs:dict={}, squeeze:bool=False, unsqueeze:bool=False):
        super().__init__()

        # assign instance variables
        self.squeeze = squeeze
        self.unsqueeze = unsqueeze
        
        # define layers
        self.encoder = MLP(
            in_features=in_features,
            out_features=embedding_size,
            **encoder_kwargs
        )
        self.decoder = MLP(
            in_features=embedding_size,
            out_features=in_features,
            **decoder_kwargs
        )
    
    def encode(self, X:Tensor, squeeze:bool=False):
        # squeeze if applicable
        if squeeze == True:
            X = X.squeeze(-1)

        # encode
        z = self.encoder(X)

        return z
    
    def decode(self, z:Tensor, unsqueeze:bool=False):
        # decode
        X_hat = self.decoder(z)

        # unsqueeze if applicable
        if unsqueeze == True:
            X_hat = X_hat.unsqueeze(-1)

        return X_hat
    
    def forward(self, X:Tensor):
        # encode, decode
        z = self.encode(X, self.squeeze)
        X_hat = self.decode(z, self.unsqueeze)

        return X_hat

#### GCN ####
    
class GraphConvLayer(nn.Module):
    def __init__(self, 
                 in_features:int, out_features:int, relation:DataFrame, num_nodes:int, 
                 adj_out:bool=False, adj_in:bool=True, adj_self:bool=True, bias:bool=True, normalize:Literal['sym','row']='row'):
        super().__init__()

        # assign instance vars
        self.out_features = out_features
        self.use_out = adj_out
        self.use_in = adj_in
        self.use_self = adj_self
        self.use_bias = bias
        
        # get adj
        if self.use_out or self.use_in:
            adj = _get_adj(relation, num_nodes, method='out')

            # assign out adj
            if self.use_out:
                self.adj_out = _norm_adj(adj, normalize)
            
            # assign in adj
            if self.use_in:
                self.adj_in = _norm_adj(adj.T, normalize)

        # assign self adj (identity)
        if self.use_self:
            self.adj_self = torch.eye(num_nodes)

        # register params
        self.weight_out = self._init_param('weight_out', self.use_out, (in_features, out_features))
        self.weight_in = self._init_param('weight_in', self.use_in, (in_features, out_features))
        self.weight_self = self._init_param('weight_self', self.use_self, (in_features, out_features))
        self.bias = self._init_param('bias', self.use_bias, (out_features,), nn.init.zeros_)

    def _init_param(self, name:str, use_param:bool, size:tuple[int, ...], init_fn:Callable[[Tensor], None]=nn.init.xavier_normal_):
        # init param if in use
        if use_param:
            param = nn.Parameter(torch.randn(*size))
            init_fn(param)
            self.register_parameter(name, param)

        # else init as None
        else:
            param = None
            self.register_parameter(name, None)

        return param

    def forward(self, X:Tensor):
        # get dims
        batch_size, num_nodes, _ = X.shape
        
        # init H
        H = torch.zeros(batch_size, num_nodes, self.out_features)

        # apply message passing, bias where applicable
        if self.use_out:
            H += self.adj_out @ X @ self.weight_out
        if self.use_in:
            H += self.adj_in @ X @ self.weight_in
        if self.use_self:
            H += self.adj_self @ X @ self.weight_self
        if self.use_bias:
            H += self.bias

        return H

class GCN(nn.Module):
    def __init__(self, 
                in_features:int, out_features:int, relation:DataFrame, num_nodes:int,
                hidden_dims:list[int]=[], act_fn=nn.LeakyReLU(), end_fn=None,  
                adj_out:bool=False, adj_in:bool=False, adj_self:bool=True, bias:bool=True, normalize:Literal['sym','row']='row'):
        super().__init__()

        # set layer kwargs
        layer_kwargs = {
            'relation':relation, 
            'num_nodes':num_nodes, 
            'adj_out':adj_out, 
            'adj_in':adj_in,
            'adj_self':adj_self,
            'bias':bias,
            'normalize':normalize
        }

        # define layers
        layers = _get_layers(
            in_features=in_features,
            out_features=out_features,
            layer_class=GraphConvLayer,
            layer_kwargs=layer_kwargs,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            end_fn=end_fn
        )

        # define model
        self.model = nn.Sequential(*layers)

    def forward(self, X:Tensor):
        return self.model(X)

class GCNClassifier(nn.Module):
    def __init__(
            self, in_features:int, out_features:int, relation:DataFrame, num_nodes:int, gcn_kwargs:dict={}, mlp_kwargs:dict={},
            adj_out:bool=False, adj_in:bool=False, adj_self:bool=True, bias:bool=True, normalize:Literal['sym','row']='row'
        ):
        super().__init__()

        # set gcn kwargs
        gcn_kwargs = {
            'adj_out':adj_out, 
            'adj_in':adj_in,
            'adj_self':adj_self,
            'bias':bias,
            'normalize':normalize,
            **gcn_kwargs
        }

        # define layers
        self.gcn = GCN(in_features=in_features, out_features=1, relation=relation, num_nodes=num_nodes, **gcn_kwargs)
        self.mlp = MLP(in_features=num_nodes, out_features=out_features, **mlp_kwargs)   

    def forward(self, X):
        # node embedding: (batch_size, num_nodes, num_features) >> (batch_size, num_nodes, 1)
        H = self.gcn(X).squeeze(-1) # squeeze >> (batch_size, num_nodes)
        
        # get logits
        logits = self.mlp(H)
        
        return logits

    def predict(self, X:Tensor, as_logits:bool=True):
        # transform if raw data (not logits)
        if as_logits == False:
            X = self.forward(X)

        # convert logits to prediction
        probs = torch.softmax(X, dim=1) # softmax to probs
        y_pred = torch.argmax(probs, dim=1) # argmax to most likely class
        y_pred = F.one_hot(y_pred, probs.shape[1]) # get one-hot encoding

        return y_pred
    
class GCNAutoencoder(nn.Module):
    def __init__(
            self, in_features:int, embedding_size:int, relation:DataFrame, num_nodes:int, 
            gcn_kwargs:dict={}, encoder_kwargs:dict={}, decoder_kwargs:dict={},
            adj_out:bool=False, adj_in:bool=False, adj_self:bool=True, bias:bool=True, normalize:Literal['sym','row']='row'
        ):
        super().__init__()

        # set gcn kwargs
        gcn_kwargs = {
            'adj_out':adj_out, 
            'adj_in':adj_in,
            'adj_self':adj_self,
            'bias':bias,
            'normalize':normalize,
            **gcn_kwargs
        }

        # define layers
        self.gcn = GCN(in_features=in_features, out_features=1, relation=relation, num_nodes=num_nodes, **gcn_kwargs)
        self.encoder = MLP(in_features=num_nodes, out_features=embedding_size, **encoder_kwargs)   
        self.decoder = MLP(in_features=embedding_size, out_features=num_nodes, **decoder_kwargs)

    def encode(self, X:Tensor):
        # X: (batch_size, num_nodes, num_features)

        # node embedding >> (batch_size, num_nodes, 1)
        H = self.gcn(X)

        # squeeze >> (batch_size, num_nodes)
        H = H.squeeze(-1)

        # encode >> (batch_size, embedding_size)
        z = self.encoder(H)

        return z
    
    def decode(self, z:Tensor):
        # z: (batch_size, embedding_size)

        # decode >> (batch_size, num_nodes)
        X_hat = self.decoder(z)

        # unqueeze >> (batch_size, num_nodes, 1)
        X_hat = X_hat.unsqueeze(-1)

        return X_hat

    def forward(self, X:Tensor):
        # encode, decode
        z = self.encode(X)
        X_hat = self.decode(z)

        return X_hat

#### GAT ####
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int, relation:DataFrame, num_nodes:int):
        super().__init__()

        # inst vars
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.leakyrelu = nn.LeakyReLU(0.2)

        # init weight 'W'
        self.weight = nn.Parameter(torch.empty(in_features, self.out_features))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

        # init weight 'a'
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # get edge indices
        self.edge_index = _get_edge_index(relation, method='in')

    def forward(self, X:Tensor):
        H = torch.matmul(X, self.weight)  # (batch_size, num_nodes, out_features)

        # Efficiently compute attention coefficients
        edge_h = torch.cat((H[:, self.edge_index[0, :]], H[:, self.edge_index[1, :]]), dim=-1)  # (batch_size, num_edges, 2*out_features)
        e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze(-1))  # (batch_size, num_edges)

        # Sparse attention using scatter
        attention = torch.zeros(X.size(0), self.num_nodes, self.num_nodes, device=X.device).fill_(-9e15)
        attention[:, self.edge_index[0], self.edge_index[1]] = e

        # Softmax norm
        attention = F.softmax(attention, dim=-1)

        # Aggregate neighbor embeddings efficiently
        H_prime = torch.matmul(attention, H)

        return H_prime
    
class GAT(nn.Module):
    def __init__(self, in_features:int, out_features:int, relation:DataFrame, num_nodes:int, hidden_dims:list=[], act_fn=nn.ELU(), end_fn=nn.ELU()):
        super().__init__()

        # define layers
        layers = _get_layers(
            in_features=in_features,
            out_features=out_features,
            layer_class=GraphAttentionLayer,
            layer_kwargs={
                'relation':relation,
                'num_nodes':num_nodes,
            },
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            end_fn=end_fn
        )

        # define model
        self.model = nn.Sequential(*layers)

    def forward(self, X:Tensor):
        return self.model(X)
    
class GATClassifier(nn.Module):
    def __init__(self, in_features:int, out_features:int, relation:DataFrame, num_nodes:int, gat_kwargs:dict={}, mlp_kwargs:dict={}):
        super().__init__()

        # define layers
        self.gat = GAT(in_features=in_features, out_features=1, relation=relation, num_nodes=num_nodes, **gat_kwargs)
        self.mlp = MLP(in_features=num_nodes, out_features=out_features, **mlp_kwargs)

    def forward(self, X):
        # node embedding: (batch_size, num_nodes, num_features) >> (batch_size, num_nodes, 1)
        H = self.gat(X).squeeze(-1) # squeeze >> (batch_size, num_nodes)
        
        # get logits
        logits = self.mlp(H)
        
        return logits

    def predict(self, X:Tensor, as_logits:bool=True):
        # transform if raw data (not logits)
        if as_logits == False:
            X = self.forward(X)

        # convert logits to prediction
        probs = torch.softmax(X, dim=1) # softmax to probs
        y_pred = torch.argmax(probs, dim=1) # argmax to most likely class
        y_pred = F.one_hot(y_pred, probs.shape[1]) # get one-hot encoding

        return y_pred

class GATAutoencoder(nn.Module):
    def __init__(self, in_features:int, embedding_size:int, relation:DataFrame, num_nodes:int, gat_kwargs:dict={}, encoder_kwargs:dict={}, decoder_kwargs:dict={}):
        super().__init__()

        # define layers
        self.gat = GAT(in_features=in_features, out_features=1, relation=relation, num_nodes=num_nodes, **gat_kwargs)
        self.encoder = MLP(in_features=num_nodes, out_features=embedding_size, **encoder_kwargs)   
        self.decoder = MLP(in_features=embedding_size, out_features=num_nodes, **decoder_kwargs)

    def encode(self, X:Tensor):
        # X: (batch_size, num_nodes, num_features)

        # node embedding >> (batch_size, num_nodes, 1)
        H = self.gat(X)

        # squeeze >> (batch_size, num_nodes)
        H = H.squeeze(-1)

        # encode >> (batch_size, embedding_size)
        z = self.encoder(H)

        return z
    
    def decode(self, z:Tensor):
        # z: (batch_size, embedding_size)

        # decode >> (batch_size, num_nodes)
        X_hat = self.decoder(z)

        # unqueeze >> (batch_size, num_nodes, 1)
        X_hat = X_hat.unsqueeze(-1)

        return X_hat
    
    def forward(self, X:Tensor):
        # encode, decode
        z = self.encode(X)
        X_hat = self.decode(z)

        return X_hat

