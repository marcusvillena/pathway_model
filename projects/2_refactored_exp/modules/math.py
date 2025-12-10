import torch
from torch import Tensor

## libsize
def library_size(x:Tensor, count_idx:int=0, num_nodes:int|None=None, num_features:int|None=None):
    # reshape if provided
    if isinstance(num_nodes, int) and isinstance(num_features, int):
        x = x.view(-1, num_nodes, num_features)

    # else assumes x is in (b,n,f)
    return x[:, :, count_idx].sum(dim=1) # (b,)

def libnorm_transform(x:Tensor, libscale:float=1.0, eps:float=1e-8):
    libsize = library_size(x).view(-1,1,1) # (b,1,1)
    return x / torch.clamp(libsize, min=eps) * libscale

def libnorm_inv_transform(x:Tensor, libsize:Tensor, libscale:float=1.0, eps:float=1e-8):
    libsize = libsize.view(-1,1,1)  # ensure (b,1,1)
    return x * libsize / torch.clamp(libscale, min=eps)

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