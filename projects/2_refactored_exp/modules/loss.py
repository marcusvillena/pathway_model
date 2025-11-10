import torch
import torch.nn as nn

from torch import Tensor
from typing import Literal

class NBLoss(nn.Module):
    def __init__(self, eps:float=1e-8, reduction:Literal['none', 'mean', 'sum']='mean', *args, **kwargs):
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