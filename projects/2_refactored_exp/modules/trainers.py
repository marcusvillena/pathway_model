from .loss import LossWrapper, KLDLoss, NBLoss
from .math import library_size
from .norm import Normalizer
from .train import Loader, Trainer
from torchmetrics.functional.classification import accuracy, precision, recall, f1_score, auroc
from torchmetrics.functional import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# typing
from torch import Tensor
from typing import Literal

class ClassifTrainer(Trainer):
    def __init__(
        self,
        lr: float, 
        pos_keys: str | list[str] | None = None,
        kw_keys: str | list[str] | dict[str,str] | None = None,
        loss_class: type[nn.Module] | None = nn.CrossEntropyLoss,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        early_stop: bool = False,
        stop_metric: str = 'loss',
        stop_kwargs: dict | None = None,
        *,
        metric_keys: str | list[str] | dict[str,str] | None = None,
        weight_method: Literal['none', 'balanced'] = 'balanced',
    ): 
        # set defaults
        kw_keys = {'input':'y_logits', 'target':'y'} if kw_keys is None else kw_keys
        metric_keys = {'pred':'input', 'target':'target'} if metric_keys is None else metric_keys

        # metric keys handling
        if isinstance(metric_keys, str):
            metric_keys = {metric_keys:metric_keys}
        if isinstance(metric_keys, (list, tuple, set)):
            metric_keys = {key:key for key in metric_keys}

        # multiloss compatibility: convert to dict, add batch_size,num_nodes
        if isinstance(kw_keys, str):
            kw_keys = {kw_keys:kw_keys}
        if isinstance(kw_keys, (list, tuple, set)):
            kw_keys = {key:key for key in kw_keys}
        if isinstance(kw_keys, dict):
            kw_keys.update({'batch_size':'batch_size', 'num_nodes':'num_nodes'})

        # init
        super().__init__(
            lr=lr, pos_keys=pos_keys, kw_keys=kw_keys, 
            loss_class=loss_class, loss_kwargs=loss_kwargs, optim_class=optim_class, optim_kwargs=optim_kwargs,
            early_stop=early_stop, stop_metric=stop_metric, stop_kwargs=stop_kwargs,
            metric_keys=metric_keys, weight_method=weight_method
        )

        # save to self
        self.metric_keys = metric_keys
        self.weight_method = weight_method
        
    def _init_with_loader(self, loader:Loader):
        # no weighting
        if self.weight_method == 'none':
            super()._init_with_loader(loader) # init self.loss_fn
        
        # scikitlearn approach
        elif self.weight_method == 'balanced':
            # get counts
            class_counts = loader.stats['class_counts']
            count = class_counts.sum()
            num_classes = class_counts.numel()

            # compute weight
            weight = count / (num_classes * class_counts)
            weight = weight / weight.mean() # normalize, mean = 1

            # init loss w/ weight
            self.loss_fn = LossWrapper(
                loss_fn = self.loss_class(weight=weight, **self.loss_kwargs),
                pos_keys = self.pos_keys,
                kw_keys= self.kw_keys,
            )

        # error case
        else:
            raise ValueError(f"weight_method should be in ['none','balanced'], got: {self.weight_method}")
    
    def _metric_key(self, key:str) -> str:
        metric_key = self.metric_keys.get(key)
        if metric_key is not None:
            return self.loss_fn.kw_keys.get(metric_key)
        else:
            return self.loss_fn.kw_keys.get(key)
        
    def _compute_metrics(self, batch_log):
        # get keys
        pred_key = self._metric_key('pred') # y_logits
        target_key = self._metric_key('target') # y

        # get data
        sample_id = torch.cat([batch['sample_id'] for batch in batch_log['data']])
        y_logits = torch.cat([batch[pred_key] for batch in batch_log['data']])
        y = torch.cat([batch[target_key] for batch in batch_log['data']])

        # compute metrics
        metrics = {}
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        kwargs = {
            'preds':y_logits,
            'target':y,
            'task':'multiclass',
            'num_classes':self.model.dims.num_classes
        }

        metrics['accuracy'] = accuracy(average='micro', **kwargs).item()
        metrics['precision'] = precision(average='macro', **kwargs).item()
        metrics['recall'] = recall(average='macro', **kwargs).item()
        metrics['f1'] = f1_score(average='macro', **kwargs).item()
        metrics['auroc'] = auroc(average='macro', **kwargs).item()

        # get values
        values = {
            'sample_id': sample_id.cpu().numpy(),
            pred_key: y_logits.cpu().numpy(),
        }

        return metrics, values

class ReconstrTrainer(Trainer):
    def __init__(
        self,
        lr:float, 
        pos_keys: str | list[str] | None = None, 
        kw_keys: str | list[str] | dict[str,str] | None = None,
        loss_class: type[nn.Module] | None = nn.MSELoss,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        early_stop: bool = False,
        stop_metric: str = 'loss',
        stop_kwargs: dict | None = None,
        *,
        metric_keys: str | list[str] | dict[str,str] | None = None,
        trainer_norm_class: type[Normalizer] = Normalizer,
        trainer_norm_kwargs: dict | None = None,
    ):
        # set defaults
        kw_keys = {'input':'x_t_pred', 'target':'x_t', 'x':'x'} if kw_keys is None else kw_keys
        metric_keys = {'pred':'input', 'target':'x'} if metric_keys is None else metric_keys
        trainer_norm_kwargs = {} if trainer_norm_kwargs is None else trainer_norm_kwargs
        
        # metric keys handling
        if isinstance(metric_keys, str):
            metric_keys = {metric_keys:metric_keys}
        if isinstance(metric_keys, (list, tuple, set)):
            metric_keys = {key:key for key in metric_keys}

        # multiloss compatibility: convert to dict, add batch_size,num_nodes
        if isinstance(kw_keys, str):
            kw_keys = {kw_keys:kw_keys}
        if isinstance(kw_keys, (list, tuple, set)):
            kw_keys = {key:key for key in kw_keys}
        if isinstance(kw_keys, dict):
            kw_keys.update({'batch_size':'batch_size', 'num_nodes':'num_nodes'})

        # init
        super().__init__(
            lr=lr, pos_keys=pos_keys, kw_keys=kw_keys, 
            loss_class=loss_class, loss_kwargs=loss_kwargs, optim_class=optim_class, optim_kwargs=optim_kwargs,
            early_stop=early_stop, stop_metric=stop_metric, stop_kwargs=stop_kwargs,
            metric_keys=metric_keys, trainer_norm_class=trainer_norm_class, trainer_norm_kwargs=trainer_norm_kwargs
        )

        # save to self
        self.metric_keys = metric_keys
        self.trainer_norm_class = trainer_norm_class
        self.trainer_norm_kwargs = trainer_norm_kwargs
        
    def _init_with_loader(self, loader:Loader):
        super()._init_with_loader(loader) # init self.loss_fn
        self.norm: Normalizer = self.trainer_norm_class(**self.trainer_norm_kwargs)
        self.norm.init_with_loader(loader)

    def _metric_key(self, key:str) -> str:
        metric_key = self.metric_keys.get(key)
        if metric_key is not None:
            return self.loss_fn.kw_keys.get(metric_key)
        else:
            return self.loss_fn.kw_keys.get(key)

    def _compute_metrics(self, batch_log):
        # get keys
        pred_key = self._metric_key('pred') # x_t_pred (model space)
        target_key = self._metric_key('target') # x (raw)

        # get data (in model transform space)
        sample_id = torch.cat([batch['sample_id'] for batch in batch_log['data']])
        x_pred = torch.cat([batch[pred_key] for batch in batch_log['data']])
        x = torch.cat([batch[target_key] for batch in batch_log['data']])

        # raw space -> trainer space -> flatten for metrics
        x_pred_flat = self.norm.transform(x_pred).view(-1)  
        x_flat = self.norm.transform(x).view(-1)

        # compute metrics
        metrics = {}
        mse = mean_squared_error(x_pred_flat, x_flat)
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']
        metrics['mse'] = mse.item()
        metrics['rmse'] = mse.sqrt().item()
        metrics['mae'] = mean_absolute_error(x_pred_flat, x_flat).item()
        metrics['r2'] = r2_score(x_pred_flat, x_flat).item()

        # get values
        values = {
            'sample_id': sample_id.cpu().numpy(),
            pred_key: x_pred.cpu().numpy(), # raw space
        }

        # VAE metrics
        z_mu_key = self._metric_key('z_mu') # z_mu (VAE latent)
        if batch_log['data'][0].get(z_mu_key) is not None:
            z_logvar_key = self._metric_key('z_logvar') # z_logvar (VAE latent)
            z_mu = torch.cat([batch[z_mu_key] for batch in batch_log['data']])
            z_logvar = torch.cat([batch[z_logvar_key] for batch in batch_log['data']])
            metrics['kld'] = KLDLoss()(z_mu, z_logvar).item()

        # generative metrics
        theta_key = self._metric_key('theta') # theta (generative)
        theta = self.model.theta.detach() if hasattr(self.model, 'theta') else None
        if theta is not None:
            metrics['nb'] = NBLoss()(
                x = x,
                mu = x_pred,
                theta = theta.view(1,-1)
            ).item()

            values[theta_key] = theta.detach().cpu().numpy()

        return metrics, values