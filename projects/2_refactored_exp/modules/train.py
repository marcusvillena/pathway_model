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
from .utils import reshape

from torch import Generator, Tensor
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Literal, Union

## DataloaderMean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from .utils import input_to_dict
from typing import Optional, Sequence

## NBReconTrainer
from .loss import NBLoss, UncertaintyLoss

# NBClassTrainer
from torchmetrics.functional.classification import multiclass_accuracy


# Dataloaders
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

class DataloaderMean():
    def __init__(self, loader, num_nodes, num_features):
        self.loader = loader
        self.num_nodes = num_nodes
        self.num_features = num_features

    def get_mean(self, target_class:Optional[Union[int, Sequence[int]]]=None, to_bnf:bool=True):
            num_nodes = self.num_nodes
            num_features = self.num_features
            loader = self.loader

            # init trackers
            total = torch.zeros(num_nodes * num_features)
            count = 0

            # ensure target_class is list -> tensor
            if isinstance(target_class, int): # if int (one class)
                target_class = torch.tensor([target_class])
            elif target_class is not None: # if iterable (multiclass)
                target_class = torch.tensor(list(target_class))

            # compute for all batches in loader
            for batch in loader:
                data = input_to_dict(batch)
                x = reshape(data['x'], 'b,n*f', num_nodes=num_nodes, num_features=num_features) # (b,n*f)
                labels = data['y']  # (b,)

                # apply class filtering if applicable
                if target_class is not None:
                    mask = torch.isin(labels, target_class)
                    if not mask.any():
                        continue
                    x = x[mask]

                # skip batch if empty (avoid div0)
                batch_size = x.shape[0]
                if batch_size == 0:
                    continue
                
                # add to running
                total += x.sum(dim=0)
                count += batch_size

            # ensure samples present (avoid div0); compute mean
            assert count > 0,  ValueError("No samples found for the specified class(es).")
            mean = total / count

            # format to b,n,f
            if to_bnf:
                mean = mean.view(1, -1, 1)

            return mean
    
    def plot_target_vs_global(self, target_class:Union[int, Sequence[int]], labels:Optional[Sequence[str]]=None, as_log:bool=True):
        # get means
        x_mean = self.get_mean(target_class=None).cpu().numpy().reshape(-1) # global as y
        x_recon_mean = self.get_mean(target_class=target_class).cpu().numpy().reshape(-1) # target as x

        # log if applic
        if as_log:
            x_mean = np.log(x_mean+1)
            x_recon_mean = np.log(x_recon_mean+1)
            mu_label = 'log(μ)'
        else:
            mu_label = 'μ'

        #### plot ####
        
        # get MAE
        recon_error = np.abs(x_mean - x_recon_mean)
        error_label = "Mean absolute error"

        # Normalize error for size and color
        size_min, size_max = 20, 200
        size = size_min + (size_max - size_min) * (recon_error - recon_error.min()) / (recon_error.ptp() + 1e-8)
        norm = mcolors.Normalize(vmin=recon_error.min(), vmax=recon_error.max())
        cmap = plt.colormaps['viridis']
        colors = cmap(norm(recon_error))

        # Darkened edge colors
        dark_edge_colors = colors.copy()
        dark_edge_colors[:, :3] *= 0.5
        dark_edge_colors[:, 3] = 1

        # Fit OLS model
        X = sm.add_constant(x_mean)
        model = sm.OLS(x_recon_mean, X).fit()
        intercept, slope = model.params
        r_squared = model.rsquared
        f_statistic = model.fvalue
        p_value = model.f_pvalue

        # Prepare regression line
        x_vals = np.linspace(x_mean.min(), x_mean.max(), 100)
        regression_line = intercept + slope * x_vals

        # Begin plot
        plt.figure(figsize=(10, 8))

        # Scatter plot
        sc = plt.scatter(x_mean, x_recon_mean, c=recon_error, cmap='viridis',
                        edgecolor=dark_edge_colors, alpha=0.4, s=size)

        # y = x reference line
        plt.plot(x_vals, x_vals, color='red', linestyle='--', label='Reference line (y = x)', linewidth=1)

        # Regression line
        plt.plot(x_vals, regression_line, color='orange', linestyle='-', label=f'Regression line', linewidth=1)

        # Colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label(f"{error_label}")

        # Axes labels and layout
        class_name = labels[target_class] if labels is not None else f'class {target_class}' # class name if prov
        plt.xlabel(f"{mu_label} of {class_name} (per gene)")
        plt.ylabel(f"Global {mu_label} (per gene)")
        plt.axis("equal")

        # Add R², F, and regression equation
        reg_eq = f"$y = {intercept:.3f} + {slope:.3f}x$"
        r2_text = f"$R^2$ = {r_squared:.3f}"
        f_text = f"$F$ = {f_statistic:.1e}"
        p_text = f"$p$ = {p_value:.1e}" if p_value > 0 else "$p$ < 1e-308"

        annotation_text = f"{reg_eq}\n{r2_text}\n{f_text}\n{p_text}"

        plt.text(0.95, 0.05, annotation_text,
                transform=plt.gca().transAxes,
                fontsize=12, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))

        # Legend for lines
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

# Training
class Trainer():
    def __init__(self, model, loss_fn:nn.Module, optimizer_class:optim.Optimizer=optim.Adam, optimizer_kwargs:dict={}, report_metrics=['loss'], verbose:bool=False):
        # assign inst vars
        self._orig_model = model
        self.loss_fn = loss_fn
        self.report_metrics = report_metrics
        self.verbose = verbose
        
        # define optimizer, paired with self.model
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs

    def run(self, loader:Loader, num_epochs:int):
        # initiate loader, predefined model & optimizer
        self.loader = loader
        self.model = self._orig_model.clone() if hasattr(self._orig_model, 'clone') else copy.deepcopy(self._orig_model)
        self.optimizer = self._optimizer_class(self.model.parameters(), **self._optimizer_kwargs)

        # verbose, use tqdm
        if self.verbose == True:
            pbar = tqdm(range(num_epochs))
        else:
            pbar = range(num_epochs)

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
                loss.backward()
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

class NBReconTrainer(Trainer):
    def _compute_loss(self, batch):
        # extract x
        data = input_to_dict(batch)
        x = data.get('x')

        # forward pass
        out = self.model(batch)

        # get params
        lfc = out.get('lfc') # mse
        lfc_recon = out.get('lfc_recon') # mse
        x_recon = out.get('x_recon')
        mu = out.get('mu') # nb
        theta = out.get('theta')

        log2_x = torch.log2(x + 1)
        log2_x_recon = torch.log2(x_recon + 1)

        # compute loss
        recon_loss = self.loss_fn(log2_x, log2_x_recon)
        lfc_loss = self.loss_fn(lfc, lfc_recon)
        nb_loss = NBLoss()(x, mu, theta)
        
        criterion = UncertaintyLoss(num_tasks=3)
        loss = criterion([lfc_loss, recon_loss, nb_loss])
        # loss = lfc_loss

        return loss, out
    
    def _compute_metrics(self, batch_log:dict): # change in child
        # init
        metrics = {}

        # compute loss
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        # get outputs
        x = torch.cat([batch['x'] for batch in batch_log['batch']])
        y = torch.cat([batch['y'] for batch in batch_log['batch']])
        x_recon = torch.cat([batch['x_recon'] for batch in batch_log['out']])
        mu = torch.cat([batch['mu'] for batch in batch_log['out']])
        theta = torch.cat([batch['theta'] for batch in batch_log['out']])
        lfc = torch.cat([batch['lfc'] for batch in batch_log['out']])
        lfc_recon = torch.cat([batch['lfc_recon'] for batch in batch_log['out']])

        # scale outputs to log2        
        log2_x = torch.log2(x + 1)
        log2_x_recon = torch.log2(x_recon + 1)
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
            'y': y.cpu().numpy(),
            'x': x.cpu().numpy(),
            'x_recon': x_recon.cpu().numpy(),
            'mu': mu.cpu().numpy(),
            'theta': theta.cpu().numpy(),
            'lfc':lfc.cpu().numpy(),
            'lfc_recon':lfc_recon.cpu().numpy()
        }

        return metrics, values

class NBClassTrainer(Trainer):
    def _compute_loss(self, batch):
        # extract y
        data = input_to_dict(batch)
        y = data.get('y')

        # forward pass
        out = self.model(batch)

        # get params
        logits = out.get('y_logits')


        # compute loss
        loss = self.loss_fn(logits, y)

        if self.model.nb:
            x = data.get('x')
            mu = out.get('mu') # nb
            theta = out.get('theta')
            nb_loss = NBLoss()(x, mu, theta)
            criterion = UncertaintyLoss(num_tasks=2)

            loss = criterion([loss, nb_loss])

        return loss, out

    def _compute_metrics(self, batch_log):
        # init
        metrics = {}
        values = {}

        # compute loss
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        # get outputs
        x = torch.cat([batch['x'] for batch in batch_log['batch']])
        mu = torch.cat([batch['mu'] for batch in batch_log['out']])
        theta = torch.cat([batch['theta'] for batch in batch_log['out']])

        y = torch.cat([batch['y'] for batch in batch_log['batch']])
        # y_logits = torch.cat([batch['y_logits'] for batch in batch_log['out']])
        # y_probs = torch.cat([batch['y_probs'] for batch in batch_log['out']])
        y_preds = torch.cat([batch['y_preds'] for batch in batch_log['out']])

        # compute metrics
        # metrics['accuracy'] = multiclass_accuracy(y_preds, y, self.model.dims.num_classes, average='none').item()
        metrics['micro_acc'] = multiclass_accuracy(y_preds, y, self.model.dims.num_classes, average='micro').item()
        metrics['macro_acc'] = multiclass_accuracy(y_preds, y, self.model.dims.num_classes, average='macro').item()
        # metrics['weighted_acc'] = multiclass_accuracy(y_preds, y, self.model.dims.num_classes, average='weighted').item() # identical to micro for accuracy

        values = {
            'y': y.cpu().numpy(),
            'x': x.cpu().numpy(),
            'mu': mu.cpu().numpy(),
            'theta': theta.cpu().numpy(),
            'y_preds': y_preds.cpu().numpy()
        }

        
        return metrics, values

# Experiment