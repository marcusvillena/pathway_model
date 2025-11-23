# general
import torch
import torch.nn as nn
from .utils import capture_kwargs, clean_name, filter_kwargs, input_to_dict, reshape

# loader
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# trainer
from tqdm.auto import tqdm
import copy
import inspect
import torch.optim as optim

# experiment
from datetime import datetime
from pathlib import Path
from scipy import stats
import json
import pandas as pd
import numpy as np

# grid 
import itertools

# trainer metrics
from torchmetrics.functional.classification import accuracy, precision, recall, f1_score, auroc

# typing
from .data import DataWrapper
from torch import Generator, Tensor
from typing import Literal, Optional, Sequence, Union

## Main

class Loader():
    def __init__(self, dataset:DataWrapper, device:str, batch_size:int=16, val_size:int=0.15, test_size:int=0.15, stats_kwargs:dict=None):
        # init seed, generator
        self.seed = torch.seed()
        generator = torch.Generator(device=device).manual_seed(self.seed)

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

        # get stats
        stats_kwargs = {} if stats_kwargs is None else stats_kwargs
        self.stats = self.get_stats(loader=self.train_loader, **stats_kwargs)

    def get_stats(self, loader:DataLoader, target_class:Optional[Union[int, Sequence[int]]]=None):
        # dims
        num_nodes = self.dataset[0].num_nodes
        num_features = self.dataset[0].num_features
        num_classes = self.dataset.num_classes

        # init trackers
        sum_x = torch.zeros(num_nodes, num_features)
        sum_x2 = torch.zeros(num_nodes, num_features)
        sum_logx = torch.zeros(num_nodes, num_features)
        sum_logx2 = torch.zeros(num_nodes, num_features)
        count = 0
        class_counts = torch.zeros(num_classes, dtype=torch.long)


        # ensure target_class is list -> tensor
        if isinstance(target_class, int): # if int (one class)
            target_class = torch.tensor([target_class])
        elif target_class is not None: # if iterable (multiclass)
            target_class = torch.tensor(list(target_class))

        # compute for all batches in loader
        for batch in loader:
            data = input_to_dict(batch)
            x = reshape(data['x'], 'b,n,f', num_nodes=num_nodes, num_features=num_features) # (b,n,f)
            y = data['y']  # (b,)

            # apply class filtering if applicable
            if target_class is not None:
                mask = torch.isin(y, target_class)
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
            class_counts += torch.bincount(y, minlength=class_counts.numel())

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
            'log_std': log_std,
            'class_counts': class_counts
        }

class Trainer():
    def __init__(
        self, 
        lr:float, 
        loss_class:type[nn.Module], 
        loss_kwargs:dict|None = None, 
        optim_class:type[optim.Optimizer] = optim.Adam, 
        optim_kwargs:dict|None = None, 
        report_metrics:list[str]|None = None,
        **kwargs 
    ):
        # get kwargs
        sig = inspect.signature(type(self).__init__)
        self._orig_kwargs = capture_kwargs(sig, self, lr, loss_class, loss_kwargs, optim_class, optim_kwargs, report_metrics, **kwargs)

        # defaults
        self.lr = lr
        self.loss_class = loss_class
        self.loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        self.report_metrics = ['loss'] if report_metrics is None else report_metrics

    def run(self, model:nn.Module, loader:Loader, num_epochs:int, verbose:bool=False):
        # get loader
        self.loader = loader

        # initiate loss
        self.loss_fn = self._init_loss(self.loader)

        # initiate model
        self.model = model.clone() if hasattr(model, 'clone') else copy.deepcopy(model)
        if hasattr(self.model, 'init_with_loader'):
            self.model.init_with_loader(self.loader)

        # init optimizer (requires model)
        self.optimizer = self.optim_class(self.model.parameters(), lr=self.lr, **self.optim_kwargs)        

        # train, val loop
        self.dev_metrics = {}

        pbar = tqdm(range(num_epochs), leave=verbose)
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
            epoch_report = f'Epoch {epoch+1}/{num_epochs}, Train: {train_report},    Val: {val_report}'
            pbar.set_postfix_str(epoch_report)

        # test
        self.test_metrics, self.test_values = self._run_phase('eval', self.loader.test_loader)

        # print test report
        self.test_report = self._generate_report(self.test_metrics, self.report_metrics)
        if verbose:
            tqdm.write(f'Test\t {self.test_report}\n')

        # mark model as trained
        if hasattr(self.model, 'is_trained'):
            self.model.is_trained = True

    def _generate_report(self, metrics:dict, report_metrics:list):
        # generate report
        report = (4*' ').join(
            f'{metric}={metrics[metric]:.4f}'
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

    #### change in child objects: ####

    def _init_loss(self, loader:Loader):
        return self.loss_class(**self.loss_kwargs)

    def _compute_loss(self, batch): # change in child
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

class Experiment():
    def __init__(
        self,
        device:str,
        num_trials:int,
        num_epochs:int,
        dataset:DataWrapper,
        batch_size:int=16,
        val_size:int=0.15,
        test_size:int=0.15,
    ): 
        self.device = device
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

        # init configs
        self.models = {}
        self.configs = {}

    def add_config(self, name:str, model:nn.Module, trainer:Trainer):
        # ensure name is string
        name = str(name)

        # generate unique name if duplicates
        if name in self.configs:

            # loop to find unique name
            i = 0
            candidate = f'{name}_{i}'

            while candidate in self.configs:
                i+=1
                candidate = f'{name}_{i}'
            
            # print warning, set candidate as new name
            print(f'Warning: {name} already in configs, adding as {candidate}')
            name = candidate
            
        # add to dicts
        self.models[name] = model
        self.configs[name] = trainer

    def add_grid(self, model_grid:nn.Module|dict[str,nn.Module], trainer_grid:Trainer|dict[str,Trainer]):
        # convert to dict if instance provided
        if isinstance(model_grid, nn.Module):
            model_grid = {'':model_grid}
        if isinstance(trainer_grid, Trainer):
            trainer_grid = {'':trainer_grid}

        # iterate over grids
        for model_name, model in model_grid.items():
            for trainer_name, trainer in trainer_grid.items():
                
                # build name; fallback as 'config' if both == ''
                name_parts = []
                if model_name != '': name_parts.append(model_name) 
                if trainer_name != '': name_parts.append(trainer_name)
                name = '_'.join(name_parts) if name_parts else 'config'

                # add config
                self.add_config(name, model, trainer)

    def run_experiment(self, comment:str=None, save_csv:bool=False, save_params:bool=False, save_model:bool=False, verbose:bool=True, loader_kwargs:dict=None):
        loader_kwargs = {} if loader_kwargs is None else loader_kwargs

        # make folder
        self.folder = self._get_folder(comment) if True in (save_csv, save_params, save_model) else None
 
        # init trackers
        dev_metrics = {}
        test_metrics = {}
        seeds = []

        # trial loop
        trial_pbar = tqdm(range(self.num_trials), leave=verbose)
        trial_pbar.set_postfix_str(f'Experiment {comment}')
        report = f'Trial 1/{self.num_trials}'
        for trial in trial_pbar:

            # init loader per trial
            trial_loader = Loader(
                device=self.device,
                dataset=self.dataset,
                batch_size=self.batch_size,
                val_size=self.val_size,
                test_size=self.test_size,
                **loader_kwargs
            )

            # add trial to trackers
            dev_metrics[trial] = {}
            test_metrics[trial] = {}
            seeds.append(trial_loader.seed)

            # for each config
            config_pbar = tqdm(self.configs.items(), leave=False)
            for i, (config_name, config) in enumerate(config_pbar):
                # update pbar
                report = f'Trial {trial+1}/{self.num_trials}, Config: {i+1}/{len(self.configs)} ({config_name})'
                config_pbar.set_postfix_str(report)
    
                # run config with trial loader
                config.run(model=self.models[config_name], loader=trial_loader, num_epochs=self.num_epochs)

                # add trial config to trackers
                dev_metrics[trial][config_name] = config.dev_metrics
                test_metrics[trial][config_name] = config.test_metrics

                # save
                if save_params or save_model:
                    subfolder = self.folder / f'{config_name}'
                    subfolder.mkdir(parents=True, exist_ok=True)

                if save_params:
                    config_params = {'model': config.model._orig_kwargs, 'trainer': config._orig_kwargs}
                    with open(subfolder / f'{config_name}_params.json', 'w') as f:
                        json.dump(config_params, f, indent=4, default=str)

                if save_model:
                    state_dict = config.model.state_dict()
                    torch.save(state_dict, subfolder / f'{config_name}_trial_{trial}_model.pth')

                # print report
                if verbose:
                    tqdm.write(f'{report},\t {config.test_report}')
                
            # save metrics, params per trial
            self.dev_df = self._format_outputs(dev_metrics, 'dev', save_csv)
            self.test_df = self._format_outputs(test_metrics, 'test', save_csv)
            self._save_params(seeds, save_params)
        
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
        # mean, std, ci if trials > 1
        if self.num_trials > 1:
            # ci helper fxn
            def _get_ci(series:pd.Series, confidence:float=0.95):
                n = series.count()
                sem = stats.sem(series, nan_policy='omit')
                ci = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
                return ci

            # group df, get summary stats
            summary_df = df.groupby(['config','metric'])['value'].agg(mean='mean', std='std', ci=_get_ci).reset_index()

        # 1 trial only, std and ci can't be calculated
        else:
            summary_df = df.groupby(['config','metric'])['value'].agg(mean='mean').reset_index()

        # save csv
        if save_csv:
            summary_df.to_csv(self.folder / f'summary.csv', index=False)

        return summary_df

    def _save_params(self, seeds:list[int]|None, save_params:bool=False):
        if save_params:
            # get params
            counts_params = self.dataset.wrapper.counts_params
            edge_params = self.dataset.wrapper.edge_params
            pathway_params = self.dataset.wrapper.pathway_params
            expt_params = {
                'num_trials':self.num_trials,
                'num_epochs':self.num_epochs,
                'batch_size':self.batch_size,
                'val_size':self.val_size,
                'test_size':self.test_size,
                'seeds':seeds
            }

            # collect to dict
            params = {
                'counts': counts_params,
                'edge': edge_params,
                'pathway': pathway_params,
                'experimetn': expt_params
            }

            # save
            with open(self.folder / f'params.json', 'w') as f:
                json.dump(params, f, indent=4, default=str)


def grid(obj, *, prefix:str=None, suffix:str=None, **param_lists):
    # init model dict
    objs = {}

    # get keys (each param)
    keys = list(param_lists.keys())

    # filter kwargs (safe func/class.__init__)
    obj = filter_kwargs(obj)

    # for each value combination
    for values in itertools.product(*param_lists.values()):
        # get params
        params = dict(zip(keys,values))

        # clean & build name
        name = '_'.join([f'{clean_name(k)}{clean_name(v)}' for k,v in params.items()])

        # add prefix, suffix
        if prefix is not None:
            name = f'{prefix}_{name}'
        if suffix is not None:
            name = f'{name}_{suffix}'

        # build config name & model
        objs[name] = obj(**params)

    return objs

## Trainers

class ClassifTrainer(Trainer):
    def __init__(
        self, 
        lr:float, 
        loss_class:type[nn.Module],
        loss_kwargs:dict|None = None, 
        optim_class:type[optim.Optimizer] = optim.Adam, 
        optim_kwargs:dict|None = None, 
        report_metrics:list[str]|None = None, 
        *,
        weight_method:Literal['none', 'balanced']|None = None,
    ):
        super().__init__(lr, loss_class, loss_kwargs, optim_class, optim_kwargs, report_metrics, weight_method=weight_method)
        self.weight_method = 'none' if weight_method is None else weight_method
    
    def _init_loss(self, loader:Loader):
        # get counts
        class_counts = loader.stats['class_counts']
        count = class_counts.sum()
        num_classes = class_counts.numel()

        # compute class weight
        if self.weight_method == 'none': # no weighting
            weight = None

        elif self.weight_method == 'balanced': # scikitlearn approach
            weight = count / (num_classes * class_counts) 
            weight = weight / weight.mean() # normalize, mean = 1

        else:
            raise ValueError(f"weight_method should be in ['none','balanced'], got: {self.weight_method}")

        # if weights needed externally
        self.class_weight = weight

        # construct loss function
        if self.class_weight is None:
            return self.loss_class(**self.loss_kwargs)
        else:
            return self.loss_class(weight=self.class_weight, **self.loss_kwargs)

    def _compute_loss(self, batch):
        # extract y
        data = input_to_dict(batch)
        y = data.get('y')

        # forward pass
        out = self.model(batch)
        logits = out.get('y_logits')

        # compute loss
        loss = self.loss_fn(logits, y)

        return loss, out
    
    def _compute_metrics(self, batch_log):
        # init
        metrics = {}

        # compute loss
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        # get outputs
        x = torch.cat([batch['x'] for batch in batch_log['batch']])
        y = torch.cat([batch['y'] for batch in batch_log['batch']])
        y_logits = torch.cat([batch['y_logits'] for batch in batch_log['out']])

        # compute metrics
        kwargs = {
            'preds':y_logits,
            'target':y,
            'task':'multiclass',
            'num_classes':self.model.dims.num_classes
        }
        metrics['accuracy'] = accuracy(average='micro', **kwargs).item()
        metrics['precision'] = precision(average="macro", **kwargs).item()
        metrics['recall'] = recall(average='macro', **kwargs).item()
        metrics['f1'] = f1_score(average="macro", **kwargs).item()
        metrics['auroc'] = auroc(average="macro", **kwargs).item()

        # get values
        values = {
            'x': x.cpu().numpy(),
            'y': y.cpu().numpy(),
            'y_logits': y_logits.cpu().numpy(),
        }

        return metrics, values


##
