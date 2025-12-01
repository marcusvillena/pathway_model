from .loss import LossWrapper
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
        pos_keys: str | list[str] = None, # defaults to ('y_logits','y')
        out_keys: str | list[str] | dict[str,str] | None = 'y_logits',
        batch_keys: str | list[str] | dict[str,str] | None = 'y',
        loss_class: type[nn.Module] | None = nn.CrossEntropyLoss,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        *,
        weight_method: Literal['none', 'balanced'] = 'balanced',
    ): 
        # set defaults
        pos_keys = ('y_logits','y') if pos_keys is None else pos_keys

        # init
        super().__init__(
            lr, pos_keys, out_keys, batch_keys, loss_class, loss_kwargs, optim_class, optim_kwargs, 
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
        preds_key = self.loss_fn.extra_keys[self.pos_keys[0]]
        target_key = self.loss_fn.extra_keys[self.pos_keys[1]]

        # get data
        sample_id = torch.cat([batch['sample_id'] for batch in batch_log['data']])
        y_logits = torch.cat([batch[preds_key] for batch in batch_log['data']])
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
            preds_key: y_logits.cpu().numpy(),
        }

        return metrics, values

class ReconstrTrainer(Trainer):
    def __init__(
        self,
        lr:float, 
        pos_keys: str | list[str] = None, # defaults to ('x_preds', 'x_t', 'x')
        out_keys: str | list[str] | dict[str,str] | None = ('x_preds', 'x_t'), # both from model (transformed for loss)
        batch_keys: str | list[str] | dict[str,str] | None = 'x',
        loss_class: type[nn.Module] | None = nn.MSELoss,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        *,
        norm_class: type[Normalizer] = Normalizer,
        norm_kwargs: dict | None = None,
        target_key: str = 'x', # to get 'x' out of batch for metrics
    ):
        # set defaults
        pos_keys = ('x_preds','x_t') if pos_keys is None else pos_keys
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        
        # init
        super().__init__(
            lr, pos_keys, out_keys, batch_keys, loss_class, loss_kwargs, optim_class, optim_kwargs,
            norm_class=norm_class, norm_kwargs=norm_kwargs, target_key=target_key
        )

        # save to self
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs
        self.target_key = target_key
        
    def _init_with_loader(self, loader:Loader):
        super()._init_with_loader(loader) # init self.loss_fn
        self.norm: Normalizer = self.norm_class(**self.norm_kwargs)
        self.norm.init_with_loader(loader)

    def _compute_metrics(self, batch_log):
        # get keys
        preds_key = self.loss_fn.extra_keys[self.pos_keys[0]] # x_preds (model space)
        target_key = self.loss_fn.extra_keys[self.target_key] # x (raw)

        # get data (in model transform space)
        sample_id = torch.cat([batch['sample_id'] for batch in batch_log['data']])
        x_preds = torch.cat([batch[preds_key] for batch in batch_log['data']])
        x = torch.cat([batch[target_key] for batch in batch_log['data']])

        # model space -> raw space
        x_preds_raw: Tensor = self.model.encoder.norm.inverse_transform(x_preds)

        # raw space -> trainer space
        x_preds = self.norm.transform(x_preds_raw)
        x = self.norm.transform(x)

        # flatten for metrics
        x_preds = x_preds.view(-1)
        x = x.view(-1)

        # compute metrics
        metrics = {}
        mse = mean_squared_error(x_preds, x)
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']
        metrics['mse'] = mse.item()
        metrics['rmse'] = mse.sqrt().item()
        metrics['mae'] = mean_absolute_error(x_preds, x).item()
        metrics['r2'] = r2_score(x_preds, x).item()

        # get values
        values = {
            'sample_id': sample_id.cpu().numpy(),
            f'{preds_key}_raw': x_preds_raw.cpu().numpy(), # raw space
        }

        return metrics, values