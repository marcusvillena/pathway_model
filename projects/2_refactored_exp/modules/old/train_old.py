import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from ..data import GraphDataset
from ..utils import reshape

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
from ..utils import input_to_dict
from typing import Optional, Sequence

## NBReconTrainer
from ..loss import NBLoss, UncertaintyLoss

# NBClassTrainer
from torchmetrics.functional.classification import multiclass_accuracy


# Dataloaders
class Loader():
    def __init__(self, dataset:GraphDataset, generator:Generator, batch_size:int=16, val_size:int=0.15, test_size:int=0.15, **mean_kwargs):
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

        # get mean
        self.stats = self.get_stats(**mean_kwargs)

    def get_stats(self, target_class:Optional[Union[int, Sequence[int]]]=None):
            num_nodes = self.dataset[0].num_nodes
            num_features = self.dataset[0].num_features
            loader = self.train_loader

            # init trackers
            sum_x = torch.zeros(num_nodes, num_features)
            sum_x2 = torch.zeros(num_nodes, num_features)
            sum_logx = torch.zeros(num_nodes, num_features)
            sum_logx2 = torch.zeros(num_nodes, num_features)
            count = 0

            # ensure target_class is list -> tensor
            if isinstance(target_class, int): # if int (one class)
                target_class = torch.tensor([target_class])
            elif target_class is not None: # if iterable (multiclass)
                target_class = torch.tensor(list(target_class))

            # compute for all batches in loader
            for batch in loader:
                data = input_to_dict(batch)
                x = reshape(data['x'], 'b,n,f', num_nodes=num_nodes, num_features=num_features) # (b,n,f)
                labels = data['y']  # (b,)

                # apply class filtering if applicable
                if target_class is not None:
                    mask = torch.isin(labels, target_class)
                    if not mask.any():
                        continue
                    x = x[mask]

                # skip batch if empty (avoid div0)
                if x.size(0) == 0:
                    continue
                
                # get log
                log_x = torch.log1p(x)

                # add to running
                sum_x += x.sum(dim=0) # (n,f)
                sum_x2 += (x**2).sum(dim=0) # (n,f)
                sum_logx += log_x.sum(dim=0) # (n,f)
                sum_logx2 += (log_x**2).sum(dim=0) # (n,f)
                count += x.size(0) 


            # ensure samples present (avoid div0)
            if count == 0:
                raise ValueError("No samples found for the specified class(es).")

            # compute mean
            mean = sum_x / count
            log_mean = sum_logx / count
            
            # compute std
            var = torch.clamp((sum_x2/count) - (mean.pow(2)), min=0.0)
            log_var = torch.clamp((sum_logx2/count) - (log_mean.pow(2)), min=0.0)

            std = torch.clamp(torch.sqrt(var), min=1e-8)
            log_std = torch.clamp(torch.sqrt(log_var), min=1e-8)

            # add batch dim
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
            log_mean = log_mean.unsqueeze(0)
            log_std = log_std.unsqueeze(0)

            return {
                'mean': mean,
                'std': std,
                'log_mean': log_mean,
                'log_std': log_std
            }

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
    def __init__(self, loss_fn:nn.Module, optimizer_class:optim.Optimizer=optim.Adam, optimizer_kwargs:dict={}, report_metrics=['loss'], verbose:bool=False):
        # assign inst vars
        self.loss_fn = loss_fn
        self.report_metrics = report_metrics
        self.verbose = verbose
        
        # define optimizer, paired with self.model
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs

    def run(self, model, loader:Loader, num_epochs:int):
        # initiate loader, predefined model
        self.loader = loader
        self.model = model.clone() if hasattr(model, 'clone') else copy.deepcopy(model)
        if hasattr(self.model, 'init_with_loader'):
            self.model.init_with_loader(self.loader)

        # init optimizer
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

class ReconTrainer(Trainer):
    def _compute_loss(self, batch):
        # extract x
        data = input_to_dict(batch)
        x = data.get('x')

        # forward pass
        out = self.model(batch)

        # get params
        x_recon = out.get('x_recon')

        # log scale
        log2_x = torch.log2(x + 1)
        log2_x_recon = torch.log2(x_recon + 1)

        # compute loss
        loss = self.loss_fn(log2_x, log2_x_recon)

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
        }

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
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from scipy import stats
import torch
from pathlib import Path

from modules.data import GraphDataset
from modules.train import Loader, Trainer
from torch import Generator
from typing import Literal

class Experiment():
    def __init__(
        self,
        num_trials:int,
        num_epochs:int,
        dataset:GraphDataset,
        generator:Generator,
        batch_size:int=16,
        val_size:int=0.15,
        test_size:int=0.15,
    ):
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.generator = generator
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

        # init configs
        self.models = {}
        self.configs = {}

    def add_trainer(self, config_name:str, model, trainer:Trainer):
        self.models[config_name] = model
        self.configs[config_name] = trainer

    def run_experiment(self, comment:str=None, verbose:bool=True, get_values:bool=False, save_csv:bool=False, save_model:bool=False, save_values:bool=False, loader_kwargs:dict={}):
        # make folder
        if True in [save_csv, save_model]:
            self.folder = self._get_folder(comment)
        else:
            self.folder = None

        # if verbose, use tqdm
        if verbose == True:
            pbar_trials = tqdm(range(self.num_trials))
        else:
            pbar_trials = range(self.num_trials)

        # init trackers
        dev_metrics = {}
        test_metrics = {}
        self.test_values = {} if get_values else None

        # trial loop
        for trial in pbar_trials:

            # init loader per trial
            trial_loader = Loader(
                dataset=self.dataset,
                generator=self.generator,
                batch_size=self.batch_size,
                val_size=self.val_size,
                test_size=self.test_size,
                **loader_kwargs
            )

            # add trial to trackers
            dev_metrics[trial] = {}
            test_metrics[trial] = {}
            if get_values:
                self.test_values[trial] = {}

            # for each config
            for config_name, config in self.configs.items():

                # run config with trial loader
                config.run(model=self.models[config_name], loader=trial_loader, num_epochs=self.num_epochs)

                # add trial config to trackers
                dev_metrics[trial][config_name] = config.dev_metrics
                test_metrics[trial][config_name] = config.test_metrics
                if get_values:
                    self.test_values[trial][config_name] = config.test_values

                # make config subfolder for saving
                if save_model or save_values:
                    subfolder = self.folder / f'{config_name}'
                    subfolder.mkdir(parents=True, exist_ok=True)

                # save model
                if save_model:
                    state_dict = config.model.state_dict()
                    torch.save(state_dict, subfolder / f'{config_name}_trial_{trial}.pth')

                # save values
                if save_values:
                    # save values
                    np.savez_compressed(subfolder/f'{config_name}_trial_{trial}.npz', **config.test_values)

        # convert to df
        self.dev_df = self._format_outputs(dev_metrics, 'dev', save_csv)
        self.test_df = self._format_outputs(test_metrics, 'test', save_csv)
        if get_values:
            self.test_values = self._format_outputs(self.test_values, 'values')

        # get summary
        self.summary = self._get_summary(self.test_df, save_csv)

    def _get_folder(self, comment:str=None):
        # get date, time
        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%Hh%Mm%Ss").lower()

        # get dir name
        dir_name = f'{date}_{time}'

        if (comment != None) & (type(comment) == str): 
            dir_name = dir_name + f'_{comment}' # append comment if applicable

        # create folder
        folder = Path(f'./output/{dir_name}')
        folder.mkdir(parents=True, exist_ok=True)

        return folder
    
    def _format_outputs(self, x:dict, method:Literal['dev','test','values'], save_csv:bool=False):
        if method in ['dev', 'test']:
            # reshape rows
            if method == 'dev':
                rows = [
                    {
                        'trial': trial,
                        'config': config,
                        'epoch': epoch,
                        'stage': stage,
                        'metric': metric,
                        'value': value 
                    }
                    for trial, configs in x.items()
                    for config, epochs in configs.items()
                    for epoch, stages in epochs.items()
                    for stage, metrics in stages.items()
                    for metric, value in metrics.items()
                ]

            elif method == 'test':
                rows = [
                    {
                        'trial': trial,
                        'config': config,
                        'metric': metric,
                        'value': value
                    }
                    for trial, configs in x.items()
                    for config, metrics in configs.items()
                    for metric, value in metrics.items()
                ]

            # convert to df, write to csv
            df = pd.DataFrame(rows)

            # save csv
            if save_csv:
                df.to_csv(self.folder / f'{method}.csv', index=False)

            return df
        
        elif method == 'values':
            out = {}

            for trial, configs in x.items():
                for config_name, _variables in configs.items():
                    for _variable, value in _variables.items():
                        # initialize config dict if not already made
                        if _variable not in out:
                            out[_variable] = {}

                        # initialize var list if not already made
                        if config_name not in out[_variable]:
                            out[_variable][config_name] = []

                        # append value to list
                        out[_variable][config_name].append(value)
            
            return out

    def _get_summary(self, df:pd.DataFrame, save_csv:bool=False):
        # ci helper fxn
        def _get_ci(series:pd.Series, confidence:float=0.95):
            n = series.count()
            sem = stats.sem(series, nan_policy='omit')
            ci = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
            return ci

        # group df, get summary stats
        summary_df = df.groupby(['config','metric'])['value'].agg(mean='mean', std='std', ci=_get_ci).reset_index()

        # save csv
        if save_csv:
            summary_df.to_csv(self.folder / f'summary.csv', index=False)

        return summary_df
    
##