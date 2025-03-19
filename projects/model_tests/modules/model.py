import torch
import torch.nn as nn
import torch.nn.functional as F

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

def _normalize_adjacency(A):
    # get num_nodes
    num_nodes = A.size(0)

    # add self-loops
    A_hat = A + torch.eye(num_nodes)

    # get degree mat (diagonal)
    degree = torch.sum(A_hat, dim=-1)

    # inv degree e.g. D^(-1/2)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-10))

    # normalize adj e.g. D^(-1/2) * A_hat * D^(-1/2)
    A_norm = torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)
    
    return A_norm

def _get_adjacency(relation, num_nodes):
    # get edge information
    edge_indices = torch.tensor(relation[['idx1','idx2']].values) # (num_edges, 2)

    # compute adj
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32) # (num_nodes, num_nodes)
    adj[edge_indices[:,0], edge_indices[:,1]] = 1 # set to 1
    adj = _normalize_adjacency(adj) # normalize adj

    return adj

### MLP

class MLP(nn.Module):
    def __init__(self, in_features:int, out_features:int, hidden_dims:list=[], bias=False, act_fn=nn.LeakyReLU(), end_fn=None):
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
    def __init__(self, in_features:int, out_features:int, mlp_kwargs={}):
        super().__init__()

        # define layers
        self.mlp = MLP(in_features, out_features, **mlp_kwargs)

    def forward(self, X):
        logits = self.mlp(X)
        return logits

    def predict(self, X, as_logits=True):
        # transform if raw data (not logits)
        if as_logits == False:
            X = self.forward(X)

        # convert logits to prediction
        probs = torch.softmax(X, dim=1) # softmax to probs
        y_pred = torch.argmax(probs, dim=1)

        # return ohe
        y_pred_ohe = F.one_hot(y_pred, probs.shape[1])

        return y_pred_ohe

### GCN

class GraphConvLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int, relation, num_nodes, adj=None):
        super().__init__()

        # get adj
        if adj != None: # use provided adj
            self.adj = adj
        else: # compute adj
            self.adj = _get_adjacency(relation, num_nodes)

        # init learnable weights
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X):
        # message passing
        H = torch.matmul(self.adj, X) # aX = (batch_size, num_nodes, num_features)

        # transform
        H = torch.matmul(H, self.weight) # aXW = (batch_size, num_nodes, hidden_dim)

        return H
    
class GCN(nn.Module):
    def __init__(self, in_features:int, out_features:int, relation, num_nodes, hidden_dims:list=[], act_fn=nn.LeakyReLU(), end_fn=None):
        super().__init__()

        # get_adj
        self.adj = _get_adjacency(relation, num_nodes)

        # define layers
        layer_kwargs = {'relation':relation, 'num_nodes':num_nodes}
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

    def forward(self, X):
        H = self.model(X)
        return H

class GCNClassifier(nn.Module):
    def __init__(self, in_features:int, out_features:int, relation, num_nodes, gcn_kwargs={}, mlp_kwargs={}):
        super().__init__()

        # define layers
        self.gcn = GCN(in_features=in_features, out_features=1, relation=relation, num_nodes=num_nodes, **gcn_kwargs)
        self.mlp = MLP(in_features=num_nodes, out_features=out_features, **mlp_kwargs)

    def forward(self, X):
        # node embedding
        H = self.gcn(X).squeeze(-1) # (batch_size, num_nodes, num_features) >> (batch_size, num_nodes, 1) >> (batch_size, num_nodes)
        
        # get logits
        logits = self.mlp(H)
        
        return logits

    def get_embeddings(self, X):
        # node embedding
        H = self.gcn(X).squeeze(-1) # (batch_size, num_nodes, num_features) >> (batch_size, num_nodes, 1) >> (batch_size, num_nodes)

        return H
    
    def predict(self, X, as_logits=True):
        # transform if raw data (not logits)
        if as_logits == False:
            X = self.forward(X)

        # convert logits to prediction
        probs = torch.softmax(X, dim=1) # softmax to probs
        y_pred = torch.argmax(probs, dim=1)

        # return ohe
        y_pred_ohe = F.one_hot(y_pred, probs.shape[1])

        return y_pred_ohe