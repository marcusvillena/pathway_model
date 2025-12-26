import torch
import torch.nn as nn

from .utils import filter_kwargs, input_to_dict
from torch import Tensor
from typing import Any, Literal

class LossWrapper(nn.Module):
    def __init__(
        self, 
        loss_fn:nn.Module, 
        pos_keys:str|list[str]|None = None,
        out_keys:str|list[str]|dict[str,str]|None = None,
        batch_keys:str|list[str]|dict[str,str]|None = None
    ):
        '''
        pos_keys: str or list[str] of x_key names passed to loss_fn
        out_keys/batch_keys: dict[str,str] maps of {loss_arg_name: x_key}
        str and list[str] will assumes loss_arg_name == x_key.
        '''
        super().__init__()
        self.loss_fn = loss_fn
        pos_keys = [] if pos_keys is None else pos_keys
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

def reduce_loss(x:Tensor, reduction:Literal['none','sum','mean']='mean', dim:int|list[int]|None=None):
    # reduces/aggregates loss output
    if reduction == 'none':
        return x 
    if reduction == 'sum':
        return x.sum(dim=dim) 
    if reduction == 'mean':
        return x.mean(dim=dim)
    else:
        raise ValueError(f"Unknown reduction method: '{reduction}'")

class MultiLoss(nn.Module):
    def __init__(
        self, 
        loss_classes: type[nn.Module] | list[type[nn.Module]],
        loss_weights: list[float] | None = None,
        loss_inputs: list[dict[str,str] | None] | None = None,
        reduction: Literal['none','sum','mean'] = 'mean',
        eps: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

        # ensure list
        if not isinstance(loss_classes, list):
            loss_classes = [loss_classes]

        # initialize loss funcs
        self.loss_fns = nn.ModuleList([filter_kwargs(loss_class)(reduction='none', eps=self.eps, **kwargs) for loss_class in loss_classes])
        self.num_losses = len(self.loss_fns)

        # initialize weights, inputs
        self.register_buffer('loss_weights', None)
        self.loss_weights = self._set_loss_weights(loss_weights)
        self.loss_inputs = self._set_loss_inputs(loss_inputs)

    def _set_loss_weights(self, loss_weights: list[float] | None):
        # equal weights if None (default)
        if loss_weights is None:
            w = torch.ones(self.num_losses, dtype=torch.float32)

        # custom weights if provided
        else:
            w = torch.as_tensor(loss_weights, dtype=torch.float32) # ensure tensor

            if w.numel() != self.num_losses: # ensure dims match
                raise RuntimeError(f'Number of loss_weights ({w.numel()}) must match number of losses ({self.num_losses})')

        # normalize to mean = 1
        w = w / w.mean()

        # save to self (buffer)
        return w

    def _set_loss_inputs(self, loss_inputs: list[dict[str,str] | None] | None):
        # default to None
        if loss_inputs is None:
            return [None] * self.num_losses
        
        # ensure dims match
        if len(loss_inputs) != self.num_losses:
            raise RuntimeError(f'Number of loss_inputs ({len(loss_inputs)}) must match number of losses ({self.num_losses})')
        
        return loss_inputs

    def _reduce_per_sample(self, losses:list[Tensor], batch_size:int, num_nodes:int|None = None) -> Tensor:
        reduced: list[Tensor] = []

        # reduce, mixed dim losses
        for term in losses:
            if term.ndim == 0: # scalar () to (B,)
                reduced.append(term.expand(batch_size))
                continue
            
            # convert (B*N, ...) to (B,N, ...) if needed
            if num_nodes is not None:
                if term.shape[0] == batch_size * num_nodes:
                    new_shape = (batch_size, num_nodes, *term.shape[1:]) if term.ndim > 1 else (batch_size, num_nodes)
                    term = term.view(new_shape)

            # ensure B is first dim
            if term.shape[0] != batch_size:
                raise ValueError(f'First dim of loss term ({term.shape[0]}) does not match batch_size ({batch_size}).')

            # already (B,) per-sample loss
            if term.ndim == 1:
                reduced.append(term) 

            # reduce (B, ...) to (B,)
            else: 
                term_dims = tuple(range(1, term.ndim)) # reduce all dims (1 to ndim) except batch (0)
                reduced.append(reduce_loss(term, reduction='mean', dim=term_dims))

        return torch.stack(reduced, dim=0)  # (num_losses, batch_size)

    def forward(self, *, batch_size:int, num_nodes:int|None = None, **kwargs):
        losses: list[Tensor] = []

        # compute losses in list of tensors [(batch_size,...), ...]
        for loss_fn, loss_input in zip(self.loss_fns, self.loss_inputs):
            if loss_input is None:
                loss_kwargs = kwargs
            else:
                loss_kwargs = {key:kwargs[value_key] for key,value_key in loss_input.items()}

            # drop batch_size, num_nodes from loss_kwargs
            loss_kwargs = {k:v for k,v in loss_kwargs.items() if k not in ('batch_size','num_nodes')}
            losses.append(filter_kwargs(loss_fn.forward)(**loss_kwargs))

        # get per-sample losses (num_losses, batch_size)
        losses: Tensor = self._reduce_per_sample(losses, batch_size, num_nodes)

        # broadcast w: loss (L,B) * w (L,1), or loss (L,) * w (L,) -> (L,B) or (L,)
        w = self.loss_weights.view(-1, *([1] * (losses.ndim - 1)))
        
        # apply weights and sum to 
        losses = (losses * w).sum(dim=0)  # (batch_size,)

        # reduce/aggregate per-sample loss
        loss = reduce_loss(losses, reduction=self.reduction)

        return loss

# experimental, not sure if did anything
class EMAMultiLoss(MultiLoss):
    def __init__(
        self, 
        loss_classes: type[nn.Module] | list[type[nn.Module]],
        loss_weights: list[float] | None = None,
        reduction: Literal['none','sum','mean'] = 'mean',
        ema_norm: bool = False,
        alpha: float = 0.01,
        eps: float = 1e-8,
        **kwargs
    ):
        super().__init__(
            loss_classes=loss_classes, 
            loss_weights=loss_weights, 
            reduction=reduction,
            eps=eps, 
            **kwargs
        )

        self.ema_norm = ema_norm
        self.alpha = alpha
        self.register_buffer('ema', None)

    def ema_norm_transform(self, loss:Tensor):
        with torch.no_grad():
            # make into broadcastable shape, (num_losses, 1)
            stat = loss.detach().view(loss.size(0), -1).mean(dim=1, keepdim=True)

            # calculate, update ema
            if self.ema is None:
                self.ema = stat
            else:
                self.ema = (1 - self.alpha) * self.ema + self.alpha * stat

        # normalize, broadcasts self.ema to loss shape (num_losses, ...)
        return loss / torch.clamp(self.ema, min=self.eps)

    def forward(self, **kwargs):
        # compute losses
        losses = [filter_kwargs(loss_fn.forward)(**kwargs) for loss_fn in self.loss_fns]
        losses = torch.stack(losses, dim=0)
        
        # normalize losses via 1/ema
        if self.ema_norm:
            losses = self.ema_norm_transform(losses)

        # apply weights
        losses = losses * self.loss_weights

        # reduce/aggregate loss
        loss = reduce_loss(losses, reduction=self.reduction)

        return loss

class KLDLoss(nn.Module):
    # KL divergence loss between Norm(mu, sigma^2) and Norm(0, I)
    # !!! consider KL annealing/scheduling !!!
    def __init__(self, warmup:int=0, reduction:Literal['none','sum','mean']='mean'):
        super().__init__()
        self.warmup = warmup # num warmup steps
        self.step = 0 # step tracker
        self.reduction = reduction

    def forward(self, mu:Tensor, logvar:Tensor) -> Tensor:
        # mu, logvar in (batch, embed)
        # per-sample loss: (batch, embed) summed to (batch,)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # reduce
        kl = reduce_loss(kl, self.reduction)

        # warmup
        if self.training and self.warmup > 0:
            kl = kl * min(1.0, self.step/self.warmup)
            self.step += 1

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