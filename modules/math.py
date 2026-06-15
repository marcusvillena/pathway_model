import torch
import numpy as np
from torch import Tensor
from .utils import assert_finite, assert_nonnegative

## trainer
class ExpMovingAverage():
    def __init__(self, alpha: float = 0.9, warmup: int = 0, eps: float = 1e-8) -> None:
        self.target_alpha = float(alpha)
        self.warmup = int(warmup)
        self.eps = float(eps)

        # trackers
        self.epoch = 0
        self.mean: float|None = None
        self.var: float|None = None

    @property
    def alpha(self) -> float:
        if self.warmup <= 0:
            return self.target_alpha
        if self.epoch < self.warmup:
            return self.target_alpha * (self.epoch + 1) / self.warmup
        return self.target_alpha
    
    @property
    def n_eff(self) -> float:
        return (1 + self.alpha) / max(1 - self.alpha, self.eps)
    
    def moe(self, z:float=1.96) -> float|None:
        if self.var is None:
            return None
        return z * np.sqrt(self.var / self.n_eff)
    
    def update(self, x:float|torch.Tensor) -> None:
        # convert to float
        if torch.is_tensor(x):
            x = x.detach().item()
        x = float(x)

        # initialize ema if None
        if self.mean is None:
            self.mean = x
            self.var = 0.0

        # update ema
        else:
            alpha = self.alpha
            mean_prev = self.mean

            self.mean = alpha * self.mean + (1 - alpha) * x
            self.var = alpha * self.var + (1 - alpha) * (x - self.mean) * (x - mean_prev)
            self.var = max(self.var, 0.0)  # ensure non-negative variance

        self.epoch += 1
        return self.mean, self.var
    

## libsize
def library_size(x:Tensor, count_idx:int=0, num_nodes:int|None=None, num_features:int|None=None):
    # reshape if provided
    if isinstance(num_nodes, int) and isinstance(num_features, int):
        x = x.view(-1, num_nodes, num_features)

    # else assumes x is in (b,n,f)
    libsize = x[:, :, count_idx].sum(dim=1) # (b,)

    return libsize

def libnorm_transform(x:Tensor, libsize:Tensor|None=None, libscale:float=1.0, eps:float=1e-8):
    if libsize is None:
        libsize = library_size(x)
    libsize = libsize.view(-1,1,1) # ensure (b,1,1)
    return x / torch.clamp(libsize, min=eps) * libscale

def libnorm_inv_transform(x:Tensor, libsize:Tensor, libscale:float=1.0, eps:float=1e-8):
    libsize = libsize.view(-1,1,1)  # ensure (b,1,1)
    out = x * libsize / torch.clamp(libscale, min=eps)
    return out

## standardize (Z-score)
def z_transform(x:Tensor, mean:Tensor, std:Tensor, eps:float=1e-8) -> Tensor:
    return (x - mean)/torch.clamp(std, min=eps)

def z_inv_transform(z:Tensor, mean:Tensor, std:Tensor) -> Tensor:
    return (z * std) + mean

## NB
def nb_theta(mean:Tensor, var:Tensor, theta_lambda:float=0.5, theta_min:float=1e-6, theta_max:float=1e6, eps:float=1e-8):
    # compute theta from mean, var (method of moments)
    theta = mean.pow(2) / torch.clamp(var - mean, min=eps)  # avoid div0

    # mask finite values
    mask = torch.isfinite(theta) & (theta > 0)

    # get global theta (median of finite thetas)
    theta_global = torch.median(theta[mask]) if mask.any() else theta.new_tensor(1.0)

    # smooth with global and clamp
    theta = (1-theta_lambda) * theta + theta_lambda * theta_global
    return torch.clamp(theta, min=theta_min, max=theta_max)  

def nbvst_transform(x:Tensor, theta:Tensor, c:float=3.0/8.0, eps:float=1e-8) -> Tensor:
    # Anscombe-like NB transform
    theta = torch.clamp(theta, min=eps)
    radicand = torch.clamp(theta * (x + c), min=eps)
    return (2.0 / torch.sqrt(theta)) * torch.asinh(torch.sqrt(radicand))

def nbvst_inv_transform(nb_x:Tensor, theta:Tensor, c:float=3.0/8.0, eps:float=1e-8) -> Tensor:
    # inverse Anscombe-like NB transform
    theta = torch.clamp(theta, min=eps)
    sinh_term = torch.sinh(nb_x * torch.sqrt(theta)/2.0)
    return (sinh_term.pow(2) / theta) - c

## ZINB
def zinb_pi(mean:Tensor, theta:Tensor, zero_count:Tensor, count:int, eps:float=1e-8) -> Tensor:
    theta = torch.clamp(theta, min=eps)
    mean = torch.clamp(mean, min=0.0)
    zero_count = zero_count.to(mean.dtype)
    count = float(count)
    
    # p(0) from obs = n_zero / n_total
    p0_obs = torch.clamp(zero_count/count, min=eps)

    # p(0) from NB
    p0_nb = (theta / (theta + mean)).pow(theta)

    # calc pi, clamp
    pi = (p0_obs - p0_nb) / torch.clamp(1.0 - p0_nb, min=eps)
    return torch.clamp(pi, min=0.0, max = 1.0 - 1e-6) # 1e-6 for float32
    
def zinbvst_var(mean:Tensor, theta:Tensor, pi:Tensor, eps:float=1e-8) -> Tensor:
    theta = torch.clamp(theta, min=eps)
    mean = torch.clamp(mean, min=0.0)
    pi = torch.clamp(pi, min=0.0, max = 1.0 - 1e-6)

    # get var, clamp
    var = mean + mean.pow(2) / torch.clamp(1.0 - pi, min=eps) * (pi + 1/theta)
    return torch.clamp(var, min=eps)