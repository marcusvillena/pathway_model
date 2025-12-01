from .utils import cloneable
import torch
import torch.nn as nn

#typing
from .train import Loader
from torch import Tensor

# parent
@cloneable
class Normalizer(nn.Module): 
    # parent class template, change in child
    def __init__(self, learnable:bool=False):
        super().__init__()
        self.learnable = learnable

        # placeholder dims
        self.num_nodes = None
        self.num_features = None
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        self.register_buffer('mu', None)
        self.register_buffer('theta', None) 

    def _check_initialized(self):
        if self.num_nodes is None or self.num_features is None:
            raise ValueError("Normalizer not initialized. Call 'init_with_loader' with a Loader to initialize.")

    def _z_transform(self, x:Tensor) -> Tensor:
        return (x - self.mean)/self.std

    def _inverse_z_transform(self, z:Tensor) -> Tensor:
        return (z * self.std) + self.mean
    
    def _reshape_and_record(self, x:Tensor) -> tuple[Tensor, torch.Size]:
        orig_shape = x.shape
        x = x.reshape(-1, self.num_nodes, self.num_features)
        return x, orig_shape

    def init_with_loader(self, loader:Loader):
        self.num_nodes = loader.num_nodes
        self.num_features = loader.num_features

    def transform(self, x) -> Tensor:
        return x
    
    def inverse_transform(self, x) -> Tensor:
        return x

# raw (no VST)
class RawCounts(Normalizer):
    pass

class zRawCounts(RawCounts):
    def init_with_loader(self, loader:Loader):
        # num_nodes, num_features
        super().init_with_loader(loader)

        # mean, std
        if self.learnable:
            self.mean = nn.Parameter(loader.stats['mean'])
            self.std = nn.Parameter(loader.stats['std'])
        else:
            self.register_buffer('mean', loader.stats['mean'])
            self.register_buffer('std', loader.stats['std'])

    def transform(self, x:Tensor):
        self._check_initialized()
        x, orig_shape = self._reshape_and_record(x)
        z = self._z_transform(x)
        return z.reshape(orig_shape)
    
    def inverse_transform(self, z:Tensor):
        self._check_initialized()
        z, orig_shape = self._reshape_and_record(z)
        x = self._inverse_z_transform(z)
        return x.reshape(orig_shape)

# log VSTs
@cloneable
class logVST(Normalizer):
    def transform(self, x) -> Tensor:
        return torch.log1p(x)

    def inverse_transform(self, x) -> Tensor:
        return torch.expm1(x)
    
@cloneable
class zlogVST(logVST):
    def init_with_loader(self, loader:Loader):
        # num_nodes, num_features
        super().init_with_loader(loader)

        # mean, std (log)
        if self.learnable:
            self.mean = nn.Parameter(loader.stats['log_mean'])
            self.std = nn.Parameter(loader.stats['log_std'])
        else:
            self.register_buffer('mean', loader.stats['log_mean'])
            self.register_buffer('std', loader.stats['log_std'])

    def transform(self, x:Tensor):
        self._check_initialized()
        log_x = super().transform(x) # log1p
        log_x, orig_shape = self._reshape_and_record(log_x)
        z = self._z_transform(log_x)
        return z.reshape(orig_shape)
    
    def inverse_transform(self, z:Tensor):
        self._check_initialized()
        z, orig_shape = self._reshape_and_record(z)
        log_x = self._inverse_z_transform(z)
        x = super().inverse_transform(log_x) # expm1
        return x.reshape(orig_shape)

# NB VSTs
@cloneable
class NBVST(Normalizer):
    def __init__(self, learnable:bool=False, c:float=3.0/8.0, eps:float=1e-8):
        super().__init__(learnable)
        self.c = c
        self.eps = eps

    def init_with_loader(self, loader:Loader):
        # num_nodes, num_features
        super().init_with_loader(loader)

        # mu, theta
        theta = torch.clamp(loader.stats['theta'], min=self.eps)
        if self.learnable:
            self.mu = nn.Parameter(loader.stats['mean'])
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('mu', loader.stats['mean'])
            self.register_buffer('theta', theta)

    def transform(self, x:Tensor):
        self._check_initialized()
        x, orig_shape = self._reshape_and_record(x)

        # NB anscombe transform
        theta = torch.clamp(self.theta, min=self.eps) # safe
        radicand = torch.clamp(theta * (x + self.c), min=0.0)
        x_nb = (2.0 / torch.sqrt(theta)) * torch.asinh(torch.sqrt(radicand + self.eps))

        return x_nb.reshape(orig_shape)
    
    def inverse_transform(self, x_nb:Tensor):
        self._check_initialized()
        x_nb, orig_shape = self._reshape_and_record(x_nb)

        # inverse NB anscombe
        theta = torch.clamp(self.theta, min=self.eps) # safe
        sinh_term = torch.sinh(x_nb * torch.sqrt(theta) / 2.0)
        x = (sinh_term.pow(2) / theta) - self.c

        return x.reshape(orig_shape)

@cloneable
class zNBVST(NBVST):
    def init_with_loader(self, loader:Loader):
        # num_nodes, num_features; mu, theta
        super().init_with_loader(loader)

        # mean, std (nb)
        if self.learnable:
            self.mean = nn.Parameter(loader.stats['nb_mean'])
            self.std = nn.Parameter(loader.stats['nb_std'])
        else:
            self.register_buffer('mean', loader.stats['nb_mean'])
            self.register_buffer('std', loader.stats['nb_std'])

    def transform(self, x):
        x_nb = super().transform(x) # NB VST
        x_nb, orig_shape = self._reshape_and_record(x_nb)
        z = self._z_transform(x_nb)
        return z.reshape(orig_shape)

    def inverse_transform(self, z):
        z, orig_shape = self._reshape_and_record(z)
        x_nb = self._inverse_z_transform(z)
        x = super().inverse_transform(x_nb) # inverse NB VST
        return x.reshape(orig_shape)

# compatibility
@cloneable
class logNorm(logVST):
    pass

@cloneable
class ZlogNorm(zlogVST):
    pass

