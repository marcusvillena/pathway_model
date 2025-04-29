import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from .data import GraphDataset
from .utils import reshape_x

from torch import Generator, Tensor
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Literal, Union

class Loader():
    def __init__(self, dataset:GraphDataset, generator:Generator, batch_size:int=16, val_size:int=0.15, test_size:int=0.15):
        # format Xy as dataset
        self.dataset = dataset

        # get split sizes
        val_size = int(val_size * len(self.dataset))
        test_size = int(test_size * len(self.dataset))
        train_size = int(len(self.dataset) - val_size - test_size)

        # train test split
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size], generator=generator)

        # get dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=generator)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)

class Trainer():
    def __init__(self, model, loader:Loader, num_epochs:int, loss_fn:nn.Module, optimizer_class:optim.Optimizer=optim.Adam, optimizer_kwargs:dict={}, report_metrics=['loss'], verbose:bool=False, clip_gradients:bool=False, detect_anomaly:bool=False, autorun:bool=True):
        # assign inst vars
        self.loader = loader
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.report_metrics = report_metrics
        self.verbose = verbose
        self.clip_gradients = clip_gradients
        self.detect_anomaly = detect_anomaly

        # define model, should be predefined
        self.model = model.clone() if hasattr(model, 'clone') else copy.deepcopy(model)

        # define optimizer, paired with self.model
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        
        # run
        if autorun:
            self.run()

    def run(self, mode:Literal['classify', 'reconstruct']=None):
        self.mode = mode

        # verbose, use tqdm
        if self.verbose == True:
            pbar = tqdm(range(self.num_epochs))
        else:
            pbar = range(self.num_epochs)

        # train, val loop
        self.dev_metrics = {}

        for epoch in pbar:
            # training
            train_metrics, _ = self._run_phase('train', self.loader.train_loader)

            # validating
            val_metrics, _ = self._run_phase('eval', self.loader.val_loader)

            # record training/validation
            self.dev_metrics[epoch] = {'train': train_metrics, 'val': val_metrics}

            # get reports
            train_report = self._generate_report(train_metrics, self.report_metrics)
            val_report = self._generate_report(val_metrics, self.report_metrics)

            # update pbar with report
            if self.verbose == True:
                epoch_report = f'Epoch {epoch:<8}' + f'Train: {train_report}' + 8*' ' + f'Val: {val_report}'
                pbar.set_postfix_str(epoch_report)

        # test
        self.test_metrics, self.test_values = self._run_phase('eval', self.loader.test_loader)

        # print test report
        if self.verbose == True:
            test_report = self._generate_report(self.test_metrics, self.report_metrics)
            tqdm.write(f'Test\t {test_report}\n')

        # mark model as trained
        if hasattr(self.model, 'is_trained'):
            self.model.is_trained = True

    def _generate_report(self, metrics:dict, report_metrics:list):
        # generate report
        report = (4*' ').join(
            f'{metric}={metrics[metric]:<.4f}'
            for metric in report_metrics
            if metric in metrics
        )

        return report

    def _run_phase(self, mode:Literal['train','eval'], dataloader:DataLoader):
        # init batch_log
        batch_log = {
            'batch_idx':0,
            'num_batches':len(dataloader),
            'loss':0,
            'batch':[], # model input; used for custom metrics
            'out':[] # model output; used for custom metrics
        }

        # train mode
        if mode == 'train':
            self.model.train()
            for batch in dataloader:
                self.optimizer.zero_grad()
                loss, out = self._compute_loss(batch) 

                if self.detect_anomaly:
                    with torch.autograd.detect_anomaly():
                        loss.backward()
                else:
                    loss.backward()
                
                if self.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                batch_log = self._update_batch_log(batch_log, loss, batch, out)

            # self.batch_log = batch_log # debugging

        # eval mode
        else:
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    loss, out = self._compute_loss(batch)
                    batch_log = self._update_batch_log(batch_log, loss, batch, out)

            # self.batch_log = batch_log # debugging

        # compute metrics
        metrics, values = self._compute_metrics(batch_log)

        return metrics, values
    
    def _update_batch_log(self, batch_log:dict, loss:Tensor, batch:Union[Tensor,tuple], out:Union[Tensor,tuple]):
        # increment batch, loss
        batch_log['batch_idx'] += 1
        batch_log['loss'] += loss.item()

        # detach batch (inputs), outputs
        batch = self._detach_items(batch)
        out = self._detach_items(out)

        # append batch, outputs
        batch_log['batch'].append(batch)
        batch_log['out'].append(out)

        return batch_log
    
    def _detach_items(self, item):
        # single tensor
        if isinstance(item, Tensor):
            return item.detach()

        # list/tuple of tensors
        elif isinstance(item, (tuple, list)):
            return type(item)(self._detach_items(i) for i in item)

        # dict
        elif isinstance(item, dict):
            return {key: self._detach_items(value) for key, value in item.items()}
        
        # PyG DataBatch or other class with .x
        elif hasattr(item, 'x'):
            out = {
                'x': self._detach_items(item.x),
                'y': self._detach_items(getattr(item, 'y', None)),
                'batch_idx': self._detach_items(getattr(item, 'batch', None)),
                'batch_size': self._detach_items(getattr(item, 'batch_size', None)),
            }

            return out
        
        # fallback
        return item

    #### change in child objects if needed: ####

    def _compute_loss(self, batch): # change in child
        # default, assume 1 item/Tensor (x)
        x = y = batch

        # extract x,y if applicable
        if isinstance(batch, (tuple,list)): # tuple, list
            if len(batch) > 0:
                x = batch[0]
            if len(batch) > 1:
                y = batch[1]
        elif isinstance(batch, dict): # dict
            x = batch.get('x', batch)
            y = batch.get('y', x)
        elif hasattr(batch, 'x'): # PyG DataBath or other class with .x
            x = batch.x
            y = getattr(batch, 'y', x)

        # assign y as x if mode=reconstruct
        if self.mode == 'reconstruct':
            y = x
        
        # get model output
        out = self.model(x)

        # compute loss
        loss = self.loss_fn(out, y)

        return loss, out
    
    def _compute_metrics(self, batch_log:dict): # change in child
        # init
        metrics = {}
        values = {}

        # compute metrics
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        return metrics, values
    
class Log2ReconTrainer(Trainer):
    def _compute_metrics(self, batch_log:dict): # change in child
        # init
        metrics = {}

        # compute loss
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        # get outputs
        x = torch.cat([
            batch['x'].view(
                batch['batch_size'],
                int(batch['x'].shape[0]/batch['batch_size']),
                -1
            )
            for batch in batch_log['batch']
        ]).squeeze(-1)
        x_recon = torch.cat([batch['x_recon'] for batch in batch_log['out']]).squeeze(-1)
        mu = torch.cat([batch['mu'] for batch in batch_log['out']]).squeeze(-1)
        theta = torch.cat([batch['theta'] for batch in batch_log['out']]).squeeze(-1)

        # scale outputs to log2        
        log2_x = torch.log2(x + 1e-6)
        log2_x_recon = torch.log2(x_recon + 1e-6)
        log2fc = log2_x_recon - log2_x
        mse = mean_squared_error(log2_x_recon, log2_x)

        # compute log2 metrics
        metrics['mean'] = torch.mean(log2fc).item()
        metrics['std'] = torch.std(log2fc).item()
        metrics['mae'] = mean_absolute_error(log2_x_recon, log2_x).item()
        metrics['mse'] = mse.item()
        metrics['rmse'] = torch.sqrt(mse).item()
        metrics['r2'] = r2_score(log2_x_recon, log2_x).item()

        # convert values to numpy
        values = {
            'x': x.cpu().numpy(),
            'x_recon': x_recon.cpu().numpy(),
            'mu': mu.cpu().numpy(),
            'theta': theta.cpu().numpy(),
        }

        return metrics, values
    
class NBTrainer(Log2ReconTrainer):
    def _compute_loss(self, batch):
        # extract x
        x = reshape_x(x=batch, to='b,n*f')

        # forward pass
        out = self.model(batch)

        # get ZINB loss params
        mu = out.get('mu')
        theta = out.get('theta')

        # compute ZINB loss
        loss = self.loss_fn(x, mu, theta)

        return loss, out
    
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