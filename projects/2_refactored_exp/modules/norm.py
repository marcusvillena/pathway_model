from .utils import cloneable, input_to_dict
from .math import (
    z_transform,
    z_inv_transform,
    nbvst_transform,
    nbvst_inv_transform,
    library_size,
    libnorm_transform,
    libnorm_inv_transform,
    nb_theta,
    zinb_pi,
)
import torch
import torch.nn as nn

#typing
from .train import Loader, LoaderStats
from torch import Tensor
from typing import Callable, Sequence

@cloneable
class Normalizer(nn.Module): 
    # parent class template, change in child
    def __init__(self, libnorm:bool=False, znorm:bool=False, learnable:bool=False, target_class:int|list[int]|None=None, eps:float=1e-8):
        super().__init__()
        self.libnorm = libnorm
        self.znorm = znorm
        self.learnable = learnable
        self.target_class = target_class
        self.eps = eps

        # placeholder dims
        self.num_nodes = None
        self.num_features = None
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        self.register_buffer('theta', None) 
        self.register_buffer('pi', None)
        self.register_buffer('libscale', None)

    def _check_initialized(self):
        if self.num_nodes is None or self.num_features is None:
            raise ValueError("Normalizer not initialized. Call 'init_with_loader' with a Loader to initialize.")
    
    def _reshape_and_record(self, x:Tensor) -> tuple[Tensor, torch.Size]:
        orig_shape = x.shape
        x = x.reshape(-1, self.num_nodes, self.num_features)
        return x, orig_shape

    def init_with_loader(self, loader:Loader, transform:Callable|None=None, **update_kwargs):
        self.num_nodes = loader.num_nodes
        self.num_features = loader.num_features
        dataloader = loader.train_loader
        libscale = loader.stats['lib_median']

        # init trackers
        _stats = LoaderStats(self.num_nodes, self.num_features, transform=transform, eps=self.eps)

        for batch in dataloader:
            x = _stats.filter_batch(batch, target_class=self.target_class)

            # skip if no samples (after filtering)
            if x is None:
                continue
            
            # transforms 
            if self.libnorm:
                x = libnorm_transform(x, libscale=libscale)

            # update trackers
            _stats.update(x, **update_kwargs)

        if _stats.count == 0:
            raise ValueError("No samples found for the specified class(es).")
        
        # compute stats
        mean, _, std = _stats.compute()

        # add to self
        if self.learnable:
            self.mean = nn.Parameter(mean)
            self.std = nn.Parameter(std)
            self.theta = nn.Parameter(loader.stats['theta'])
            self.pi = nn.Parameter(loader.stats['pi'])
            self.libscale = nn.Parameter(libscale)
        else:
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
            self.register_buffer('theta', loader.stats['theta'])
            self.register_buffer('pi', loader.stats['pi'])
            self.register_buffer('libscale', libscale)

    def get_libsize(self, x:Tensor) -> Tensor:
        if self.libnorm:
            self._check_initialized()
            return library_size(x, num_nodes=self.num_nodes, num_features=self.num_features)
        else:
            return None

    def transform(self, x:Tensor, libsize:float|None=None) -> Tensor:
        x, orig_shape = self._reshape_and_record(x)

        if self.libnorm or self.znorm:
            self._check_initialized()

        if self.libnorm:
            x = libnorm_transform(x, libsize=libsize, libscale=self.libscale)

        # for child class, transform should happen here

        if self.znorm:
            x = z_transform(x, self.mean, self.std)

        return x.reshape(orig_shape)

    def inverse_transform(self, x:Tensor, libsize:float|None=None) -> Tensor:
        x, orig_shape = self._reshape_and_record(x)

        if self.libnorm or self.znorm:
            self._check_initialized()

        if self.znorm:
            x = z_inv_transform(x, self.mean, self.std)

        # for child class, transform should happen here

        if self.libnorm and libsize is not None:
            x = libnorm_inv_transform(x, libsize=libsize, libscale=self.libscale)

        return x.reshape(orig_shape)

@cloneable
class RawCounts(Normalizer):
    pass

@cloneable
class LogCounts(Normalizer):
    # DESeq2-like
    def init_with_loader(self, loader:Loader, transform:Callable=torch.log1p):
        super().init_with_loader(loader, transform)

    def transform(self, x:Tensor, libsize:float|None=None) -> Tensor:
        x, orig_shape = self._reshape_and_record(x)

        if self.libnorm or self.znorm:
            self._check_initialized()

        if self.libnorm:
            x = libnorm_transform(x, libsize=libsize, libscale=self.libscale)

        # x -> log(x)
        x = torch.clamp(x, min=self.eps)  # avoid log(0)
        x = torch.log1p(x)

        if self.znorm:
            x = z_transform(x, self.mean, self.std)

        return x.reshape(orig_shape)

    def inverse_transform(self, x:Tensor, libsize:float|None=None) -> Tensor:
        x, orig_shape = self._reshape_and_record(x)

        if self.libnorm or self.znorm:
            self._check_initialized()

        if self.znorm:
            x = z_inv_transform(x, self.mean, self.std)

        # log(x) -> x
        x = torch.expm1(x)
        x = torch.clamp(x, min=0.0)  # avoid negative counts

        if self.libnorm and libsize is not None:
            x = libnorm_inv_transform(x, libsize=libsize, libscale=self.libscale)

        return x.reshape(orig_shape)
    
@cloneable
class NBVST(Normalizer):
    def init_with_loader(self, loader:Loader, transform:Callable=nbvst_transform):
        theta = loader.stats['theta']
        super().init_with_loader(loader, transform, theta=theta)

    def transform(self, x:Tensor, libsize:float|None=None) -> Tensor:
        self._check_initialized() # theta must exist
        x, orig_shape = self._reshape_and_record(x)            

        if self.libnorm:
            x = libnorm_transform(x, libsize=libsize, libscale=self.libscale)

        # NB VST
        x = nbvst_transform(x, self.theta)

        if self.znorm:
            x = z_transform(x, self.mean, self.std)

        return x.reshape(orig_shape)

    def inverse_transform(self, x:Tensor, libsize:float|None=None) -> Tensor:
        self._check_initialized() # theta must exist
        x, orig_shape = self._reshape_and_record(x)            

        if self.znorm:
            x = z_inv_transform(x, self.mean, self.std)

        # inverse NB VST
        x = nbvst_inv_transform(x, self.theta)

        if self.libnorm and libsize is not None:
            x = libnorm_inv_transform(x, libsize=libsize, libscale=self.libscale)

        return x.reshape(orig_shape)

# ideas
    # PearsonZINBVST
        # Transform: t = (x - mean)/zinbstd
            # zinbstd <- mean + (mean^2)/(1-pi) * (pi + 1/theta)
        # Standardize: z = (t - mean(t))/std(t)

    # HybridZINBVST
        # Transform: t = T(x) / zinbstd
            # T(x) <- Anscombe-like nb(x) or log1p(x)
            # zinbstd <- mean + (mean^2)/(1-pi) * (pi + 1/theta)
        # Standardize: z = (t - mean(t))/std(t)

    # zero-inflation should probably be its own test/grid...
    # too many ways to handle it

# new class: LibraryNorm vs. CountNorm (current)
    # x_transformed = x_gi / (s_i / target)
        # x, per gene, per individual
        # normalized by s per individual (s_i) = sum(x_gi for g in G)
        # target: median library size, geometric mean lib size, etc.

    # other implementations:
        # DESeq2 -> median-of-ratios
        # edgeR -> trimmed mean of M-values (TMM)
        # scVI -> in decoder

    # library normalization would have to occur per-sample, thus in the model itself (not Loader)
    # for batch-wise stats (e.g. for initializing weights), these can go in Loader (e.g. for scVI latent libsize)


