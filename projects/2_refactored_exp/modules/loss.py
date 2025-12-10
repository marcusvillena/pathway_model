import torch
import torch.nn as nn

from .utils import filter_kwargs, input_to_dict
from torch import Tensor
from typing import Any, Literal

class LossWrapper(nn.Module):
    def __init__(
        self, 
        loss_fn:nn.Module, 
        pos_keys:str|list[str],
        out_keys:str|list[str]|dict[str,str]|None = None,
        batch_keys:str|list[str]|dict[str,str]|None = None
    ):
        '''
        pos_keys: str or list[str] of key names (loss_key) passed to loss_fn
        out_keys/batch_keys: dict[str,str] maps of {loss_arg_name: x_key}
        str and list[str] will assumes loss_arg_name == x_key.
        '''
        super().__init__()
        self.loss_fn = loss_fn
        out_keys = {} if out_keys is None else out_keys
        batch_keys = {} if batch_keys is None else batch_keys

        # format pos_keys
        self.pos_keys: list[str] = [pos_keys] if isinstance(pos_keys,str) else pos_keys

        # format out_keys
        if isinstance(out_keys, str):
            out_keys = {out_keys:out_keys}
        elif isinstance(out_keys, (list,tuple,set)):
            out_keys = {i:i for i in out_keys}
    
        # format batch_keys
        if isinstance(batch_keys, str):
            batch_keys = {batch_keys:batch_keys}
        elif isinstance(batch_keys, (list,tuple,set)):
            batch_keys = {i:i for i in batch_keys}

        # merge batch_keys, out_keys into extra_keys
        self.extra_keys: dict[str,str] = {**out_keys, **batch_keys}

        # safety checks
        if not self.extra_keys:
            raise ValueError("One of 'batch_keys' or 'out_keys' must be provided.")
        missing_keys = [key for key in self.pos_keys if key not in self.extra_keys]
        if missing_keys:
            raise ValueError(f"All 'pos_keys' must be in one of 'batch_keys' or 'out_keys'. Missing {missing_keys}")

    def forward(self, out:dict[str,Any], batch:dict[str,Any]):
        # input to dict
        out = input_to_dict(out)
        batch = input_to_dict(batch)

        # extract kwargs, return
        values = {**out, **batch}
        extra_kwargs = {key:values[value_key] for key,value_key in self.extra_keys.items()}

        # get pos args
        pos_args = []
        for key in self.pos_keys:
            try:
                pos_args.append(extra_kwargs.pop(key))
            except KeyError:
                raise KeyError(f"pos_arg {key} not in extra_kwargs. Check pos_keys/batch_keys/out_keys.")

        return filter_kwargs(self.loss_fn.forward)(*pos_args, **extra_kwargs)

def reduce_loss(x:Tensor, reduction:Literal['none','sum','mean']='mean'):
    # reduces/aggregates loss output
    if reduction == 'none':
        return x 
    if reduction == 'sum':
        return x.sum() 
    if reduction == 'mean':
        return x.mean()
    else:
        raise ValueError(f"Unknown reduction method: '{reduction}'")

class MultiLoss(nn.Module):
    def __init__(self, loss_classes:type[nn.Module]|list[type[nn.Module]], **kwargs):
        super().__init__()

        # ensure list
        if not isinstance(loss_classes, list):
            loss_classes = [loss_classes]

        # initialize loss funcs
        self.loss_fns = [filter_kwargs(loss_class)(**kwargs) for loss_class in loss_classes]

    def forward(self, *args, **kwargs):
        pass



    

class KLDLoss(nn.Module):
    # KL divergence los between Norm(mu, sigma^2) and Norm(0, I)
    # !!! consider KL annealing/scheduling !!!
    def __init__(self, reduction:Literal['none','sum','mean']='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu:Tensor, logvar:Tensor):
        # per-sample loss
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # reduce
        kl = reduce_loss(kl, self.reduction)

        return kl
    


class NBLoss(nn.Module):
    def __init__(self, eps:float=1e-8, reduction:Literal['none','sum','mean']='mean', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.reduction = reduction

    def forward(self, x:Tensor, mu:Tensor, theta:Tensor):
        '''
        NB loss (negative log likelihood of NB)
        '''
        # common terms
        log_theta_mu = torch.log(theta + mu + self.eps)
        log_theta = torch.log(theta + self.eps)
        log_mu = torch.log(mu + self.eps)

        # NB negative log likelihood
        log_nb = -(
            theta * (log_theta - log_mu) +
            x * (log_mu - log_theta_mu) +
            torch.lgamma(x + theta + self.eps) -
            torch.lgamma(theta + self.eps) -
            torch.lgamma(x + 1)
        )

        # reduce
        if self.reduction == 'mean':
            log_nb = log_nb.mean()
        elif self.reduction == 'sum':
            log_nb = log_nb.sum()

        return log_nb
    
class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        # log(sigma^2) per task, initialized to 0 (i.e., sigma = 1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses:list):
        """
        losses: Tensor of shape (num_tasks,) or (batch_size, num_tasks)
        """
        losses = torch.stack(losses)
        precision = torch.exp(-self.log_vars)            # (num_tasks,)
        weighted = precision * losses                    # (num_tasks,) or (batch_size, num_tasks)
        reg = self.log_vars                              # (num_tasks,)
        return (weighted + reg).sum(dim=-1).mean()       # sum over tasks, mean over batch if needed