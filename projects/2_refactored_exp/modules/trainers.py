from .loss import LossWrapper, KLDLoss
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
        out_keys: str | list[str] | dict[str,str] | None = None, # {'input':'y_logits'}
        batch_keys: str | list[str] | dict[str,str] | None = None, # {'target':'y'}
        loss_class: type[nn.Module] | None = nn.CrossEntropyLoss,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        early_stop: bool = False,
        stop_metric: str = 'loss',
        stop_kwargs: dict | None = None,
        *,
        weight_method: Literal['none', 'balanced'] = 'balanced',
    ): 
        # set defaults
        out_keys = {'input':'y_logits'} if out_keys is None else out_keys
        out_keys.update({'batch_size':'batch_size', 'num_nodes':'num_nodes'}) # multiloss compatibility
        batch_keys = {'target':'y'} if batch_keys is None else batch_keys

        # init
        super().__init__(
            lr, pos_keys, out_keys, batch_keys, loss_class, loss_kwargs, optim_class, optim_kwargs, 
            early_stop=early_stop, stop_metric=stop_metric, stop_kwargs=stop_kwargs,
            weight_method=weight_method
        )

        # save to self
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
                out_keys = self.out_keys,
                batch_keys = self.batch_keys
            )

        # error case
        else:
            raise ValueError(f"weight_method should be in ['none','balanced'], got: {self.weight_method}")
    
    def _compute_metrics(self, batch_log):
        # get keys
        pred_key = self.loss_fn.extra_keys.get('input')
        target_key = self.loss_fn.extra_keys.get('target')

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
        out_keys: str | list[str] | dict[str,str] | None = None, # x_t_pred, x_t, x_pred (raw)
        batch_keys: str | list[str] | dict[str,str] | None = None, # x (raw)
        loss_class: type[nn.Module] | None = nn.MSELoss,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        early_stop: bool = False,
        stop_metric: str = 'loss',
        stop_kwargs: dict | None = None,
        *,
        trainer_norm_class: type[Normalizer] = Normalizer,
        trainer_norm_kwargs: dict | None = None,
    ):
        # set defaults
        out_keys = {'input':'x_t_pred', 'target':'x_t'} if out_keys is None else out_keys # 'x_pred':'x_pred'
        out_keys.update({'batch_size':'batch_size', 'num_nodes':'num_nodes'}) # multiloss compatibility
        batch_keys = {'x':'x'} if batch_keys is None else batch_keys
        trainer_norm_kwargs = {} if trainer_norm_kwargs is None else trainer_norm_kwargs
        
        # init
        super().__init__(
            lr, pos_keys, out_keys, batch_keys, loss_class, loss_kwargs, optim_class, optim_kwargs,
            early_stop=early_stop, stop_metric=stop_metric, stop_kwargs=stop_kwargs,
            trainer_norm_class=trainer_norm_class, trainer_norm_kwargs=trainer_norm_kwargs
        )

        # save to self
        self.trainer_norm_class = trainer_norm_class
        self.trainer_norm_kwargs = trainer_norm_kwargs
        
    def _init_with_loader(self, loader:Loader):
        super()._init_with_loader(loader) # init self.loss_fn
        self.norm: Normalizer = self.trainer_norm_class(**self.trainer_norm_kwargs)
        self.norm.init_with_loader(loader)

    def _compute_metrics(self, batch_log):
        # get keys
        pred_key = self.loss_fn.extra_keys.get('input') # x_t_pred (model space)
        target_key = self.loss_fn.extra_keys.get('x') # x (raw)

        # vae
        mu_key = self.loss_fn.extra_keys.get('mu', None) # z_mu (VAE latent)
        logvar_key = self.loss_fn.extra_keys.get('logvar', None) # z_logvar (VAE latent)
        first_mu = batch_log['data'][0].get(mu_key) # use to check if VAE

        # get data (in model transform space)
        sample_id = torch.cat([batch['sample_id'] for batch in batch_log['data']])
        x_pred = torch.cat([batch[pred_key] for batch in batch_log['data']])
        x = torch.cat([batch[target_key] for batch in batch_log['data']])

        if first_mu is not None:
            mu = torch.cat([batch[mu_key] for batch in batch_log['data']])
            logvar = torch.cat([batch[logvar_key] for batch in batch_log['data']])

        # model space -> raw space
        libsize = library_size(x, num_nodes=self.model.dims.num_nodes, num_features=self.model.dims.num_node_features)
        x_pred_raw: Tensor = self.model.encoder.norm.inverse_transform(x_pred, libsize=libsize).detach()

        # raw space -> trainer space -> flatten for metrics
        x_pred = self.norm.transform(x_pred_raw).view(-1)
        x = self.norm.transform(x).view(-1)

        # compute metrics
        metrics = {}
        mse = mean_squared_error(x_pred, x)
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']
        metrics['mse'] = mse.item()
        metrics['rmse'] = mse.sqrt().item()
        metrics['mae'] = mean_absolute_error(x_pred, x).item()
        metrics['r2'] = r2_score(x_pred, x).item()

        if first_mu is not None:
            kld_fn = KLDLoss()
            metrics['kld'] = kld_fn(mu, logvar).item()

        # get values
        values = {
            'sample_id': sample_id.cpu().numpy(),
            f'{pred_key}': x_pred_raw.cpu().numpy(), # raw space
        }

        return metrics, values