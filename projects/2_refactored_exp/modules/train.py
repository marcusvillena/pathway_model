# general
from .utils import cloneable, capture_kwargs, clean_name, get_name_recursive, filter_kwargs, input_to_dict
import torch
import torch.nn as nn

# loader
from .math import nb_theta, zinb_pi, library_size
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# trainer
from .loss import LossWrapper
from .math import ExpMovingAverage
from tqdm.auto import tqdm
import copy
import inspect
import time
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
import functools

# typing
from .data import DataWrapper
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Callable, Literal, Sequence, Union

## Loader
@cloneable
class LoaderStats(nn.Module):
    def __init__(self, num_nodes:int, num_features:int, transform:Callable|None=None, eps:float=1e-8):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.transform = transform
        self.eps = eps

        # trackers (n,f); always on
        self.sum_x = torch.zeros(self.num_nodes, self.num_features)
        self.sum_x2 = torch.zeros(self.num_nodes, self.num_features)
        self.count = 0

    def filter_batch(self, batch, target_class:int|list[int]|None=None):
        data = input_to_dict(batch)
        x = data['x'].reshape(-1, self.num_nodes, self.num_features) # (b,n,f)

        # skip filtering if no target class
        if target_class is None:
            return x
        
        # apply class filtering if applicable, mask in (b,)
        y = data['y']  # (b,)
        mask = torch.isin(y, target_class.to(device=y.device, dtype=y.dtype))

        # filter, else skip if no samples
        if mask.any():
            x = x[mask]
            return x
        else:
            return None
            
    def update(self, x:Tensor, **kwargs):
        # transform if applicable
        if callable(self.transform):
            x = self.transform(x, **kwargs)

        # update sums, (n,f)
        self.sum_x += x.sum(dim=0)
        self.sum_x2 += x.pow(2).sum(dim=0)
        self.count += x.size(0)

    def compute(self):
        # compute mean, var, std in (n,f)
        mean = self.sum_x / self.count
        var = torch.clamp(self.sum_x2/self.count - mean.pow(2), min=0.0)
        std = torch.clamp(var.sqrt(), min=self.eps)

        return mean, var, std

class Loader():
    def __init__(self, dataset:DataWrapper, device:str, batch_size:int=16, val_size:int=0.15, test_size:int=0.15, eps:float=1e-8):
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

        # get dims
        self.eps = eps if isinstance(eps, float) else 1e-8
        self.num_nodes = self.dataset[0].num_nodes
        self.num_features = self.dataset[0].num_features
        self.num_classes = self.dataset.num_classes

        # get stats
        self.stats = self.get_stats(self.train_loader)

    def get_stats(self, dataloader:DataLoader):
        # count trackers
        _stats = LoaderStats(self.num_nodes, self.num_features, eps=self.eps)
        class_counts = torch.zeros(self.num_classes, dtype=torch.long)
        zero_count = torch.zeros(self.num_nodes, self.num_features, dtype=torch.long)
        lib_median = []

        # compute for all batches in loader
        for batch in dataloader:
            # extract data
            data = input_to_dict(batch)
            x = data['x'].reshape(-1, self.num_nodes, self.num_features) # (b,n,f)
            y = data['y'] # (b,)

            # update counts
            _stats.update(x)
            class_counts += torch.bincount(y, minlength=class_counts.numel())
            zero_count += (x==0).sum(dim=0)
            lib_median.append(library_size(x).to(torch.float32))

        # compute stats
        mean, var, _ = _stats.compute()
        theta = nb_theta(mean, var, eps=self.eps)
        pi = zinb_pi(mean, theta, zero_count, _stats.count, eps=self.eps)
        lib_median = torch.cat(lib_median).median()

        return {
            'theta': theta,
            'pi': pi,            
            'class_counts': class_counts,
            'lib_median': lib_median
        }

# Trainer
class EarlyStopping():
    def __init__(
        self,
        warmup: int = 50,
        patience: int = 50,
        mode: Literal['min', 'max'] = 'min',                                   
        confidence: float = 0.95,
        decay: float = 0.95,
    ):
        self.warmup = int(warmup)
        self.patience = int(patience)
        self.mode = mode

        # stats
        self.ema = ExpMovingAverage(alpha=float(decay), warmup=self.warmup)
        self.z = stats.norm.ppf(0.5 + float(confidence) / 2.0)

        # tracking variables
        self.best_epoch: int|None = None
        self.best_value: float|None = None
        self.criteria: float|None = None
        self.counter: int = 0
        self.should_stop: bool = False

        # model
        self.best_state: dict[str, Tensor]|None = None
    
    def _update_criteria(self) -> float|None:
        # recomputes criteria: best val +- moe (ci)
        moe = self.ema.moe(self.z)
        moe = 0.0 if moe is None else float(moe)

        if self.mode == 'min':
            return self.best_value - moe
        else:
            return self.best_value + moe

    def _better(self, current:float, target:float) -> bool:
        return current < target if self.mode == 'min' else current > target

    def update(self, x:float|Tensor, epoch:int) -> None:
        # convert to float
        if torch.is_tensor(x):
            x = x.detach().item()
        x = float(x)

        # update
        self.ema.update(x)

        # initialize
        if self.best_value is None or self.criteria is None:
            self.best_value = x # update best value
            self.best_epoch = epoch # update best epoch
            self.criteria = self._update_criteria() # update criteria
            self.counter = 0 # reset counter
            return

        # update best value, if better
        if self._better(x, self.best_value):
            self.best_value = x
            self.best_epoch = epoch

        # update criteria, if improved
        if self._better(x, self.criteria):
            self.criteria = self._update_criteria()
            self.counter = 0 # reset counter

        # apply patience after warmup
        elif epoch >= self.warmup:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def update_and_save(self, x:dict|float|Tensor, epoch:int, model:nn.Module, metric:str|None = None) -> None:
        # extract metric if dict
        if isinstance(x, dict):
            if metric is None:
                raise ValueError("metric must be provided when x is a dict")
            if metric not in x:
                raise KeyError(f"metric={metric!r} not found in keys={list(x.keys())}")
            x = x[metric]

        # update, best epoch etc.
        self.update(x, epoch)

        # if best epoch, save model state
        if self.best_epoch == epoch:
            self.best_state = {
                k: v.detach().cpu().clone() 
                for k, v in model.state_dict().items()
            }

class Trainer():
    def __init__(
        self, 
        lr: float,
        pos_keys: str | list[str] | None = None,
        kw_keys: str | list[str] | dict[str,str] | None = None,
        loss_class: type[nn.Module] | None = None,
        loss_kwargs: dict | None = None,
        optim_class: type[optim.Optimizer] = optim.Adam, 
        optim_kwargs: dict | None = None,
        early_stop: bool = False,
        stop_metric: str = 'loss',
        stop_kwargs: dict | None = None,
        **kwargs # pass child kwargs to capture_kwargs
    ):
        # ensure loss_class is provided
        if loss_class is None:
            raise ValueError("'loss_class' must be provided.")
        
        # ensure kw_keys is provided !!!
        if kw_keys is None:
            raise ValueError("'kw_keys' must be provided.")
        
        # get kwargs
        sig = inspect.signature(type(self).__init__)
        self._orig_kwargs = capture_kwargs(
            sig, self, 
            lr=lr, 
            pos_keys=pos_keys, 
            kw_keys=kw_keys,
            loss_class=loss_class, 
            loss_kwargs=loss_kwargs, 
            optim_class=optim_class, 
            optim_kwargs=optim_kwargs,
            early_stop=early_stop,
            stop_metric=stop_metric,
            stop_kwargs=stop_kwargs, 
            **kwargs
        )

        # defaults
        self.lr = lr
        self.pos_keys = pos_keys
        self.kw_keys = kw_keys
        self.loss_class = loss_class
        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.optim_class = optim_class
        self.optim_kwargs = {} if optim_kwargs is None else dict(optim_kwargs)
        self.early_stop = early_stop
        self.stop_metric = stop_metric
        self.stop_kwargs = {} if stop_kwargs is None else dict(stop_kwargs)

    def run(self, model:nn.Module, loader:Loader, num_epochs:int, report_metrics:str|list[str]|None=None, verbose:bool=False):
        # defaults
        if report_metrics is None:
            report_metrics = ['loss']
        elif isinstance(report_metrics, str):
            report_metrics = [report_metrics]

        # init trainer (self.loss_fn, self.norm, etc.)
        self._init_with_loader(loader)

        # initiate model
        self.model = model.clone() if hasattr(model, 'clone') else copy.deepcopy(model)
        if hasattr(self.model, 'init_with_loader'):
            self.model.init_with_loader(loader)

        # init optimizer (requires model)
        self.optimizer = self.optim_class(self.model.parameters(), lr=self.lr, **self.optim_kwargs)        

        # train, val loop
        self.dev_metrics = {}

        # early stopping
        stopper = EarlyStopping(**self.stop_kwargs)

        # start timer
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_start = time.perf_counter()

        pbar = tqdm(range(num_epochs), leave=verbose)
        for epoch in pbar:
            # training
            train_metrics, _ = self._run_phase('train', loader.train_loader)

            # validating
            val_metrics, _ = self._run_phase('eval', loader.val_loader)

            # record training/validation
            self.dev_metrics[epoch] = {'train': train_metrics, 'val': val_metrics}

            # get reports
            train_report = self._generate_report(train_metrics, report_metrics)
            val_report = self._generate_report(val_metrics, report_metrics)

            # update pbar with report
            epoch_report = f'Epoch {epoch+1}/{num_epochs}, Train: {train_report},    Val: {val_report}'
            pbar.set_postfix_str(epoch_report)

            # early stopping
            stopper.update_and_save(val_metrics, epoch, self.model, metric=self.stop_metric)
            if stopper.should_stop and self.early_stop:
                break
        
        # load best model state
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)

        # test
        self.test_metrics, self.test_values = self._run_phase('eval', loader.test_loader)

        # end timer, append to metrics
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_end = time.perf_counter()
        self.test_metrics['time'] = time_end - time_start
        self.test_metrics['num_epochs'] = epoch
        self.test_metrics['best_epoch'] = stopper.best_epoch if (stopper is not None) else None

        # print test report
        self.test_report = self._generate_report(self.test_metrics, report_metrics)
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
            'data':[] # model input/output for custom metrics
        }

        # train mode
        if mode == 'train':
            self.model.train()
            for batch in dataloader:
                self.optimizer.zero_grad()
                loss, out = self._compute_loss(batch) 
                loss.backward()
                self.optimizer.step()
                batch_log = self._update_batch_log(batch_log, loss, out, batch)

        # eval mode
        else:
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    loss, out = self._compute_loss(batch)
                    batch_log = self._update_batch_log(batch_log, loss, out, batch)

        # compute metrics
        metrics, values = self._compute_metrics(batch_log)

        return metrics, values
    
    def _compute_loss(self, batch):     
        out = self.model(batch)
        loss = self.loss_fn(out, batch)
        return loss, out
    
    def _update_batch_log(self, batch_log:dict, loss:Tensor, out:Union[Tensor,tuple], batch:Union[Tensor,tuple]):
        # increment batch, loss
        batch_log['batch_idx'] += 1
        batch_log['loss'] += loss.item()

        # detach batch (inputs), outputs
        out = self._detach_items(out)
        batch = self._detach_items(batch)

        # convert to dict
        out = input_to_dict(out, 'out')
        batch = input_to_dict(batch, 'batch')

        # merge and append
        data = {**batch, **out}
        batch_log['data'].append(data)
        return batch_log
    
    def _detach_items(self, item):
        # single tensor
        if isinstance(item, Tensor):
            return item.detach()

        # list/tuple (recursive)
        if isinstance(item, (list, tuple, set)):
            return type(item)(self._detach_items(i) for i in item)

        # dict, PyG Data/DataBatch
        if isinstance(item, (dict, Data, Batch)):
            return {key: self._detach_items(value) for key, value in item.items()}
        
        # other class with .x
        if hasattr(item, 'x'):
            return {
                'x': self._detach_items(item.x),
                'y': self._detach_items(getattr(item, 'y', None)),
                'sample_id': self._detach_items(getattr(item, 'sample_id', None))
            }
        
        # fallback
        return item

    #### change in child objects: ####

    def _init_with_loader(self, loader:Loader|None=None):
        # construct and wrap loss function
        self.loss_fn = LossWrapper(
            loss_fn = self.loss_class(**self.loss_kwargs),
            pos_keys = self.pos_keys,
            kw_keys = self.kw_keys,
        )
    
    def _compute_metrics(self, batch_log:dict): # change in child
        # init
        metrics = {}
        values = {}

        # compute metrics
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        # get values
        sample_id = torch.cat([batch['sample_id'] for batch in batch_log['data']])
        values = {'sample_id':sample_id}

        return metrics, values

class MultiTrainerStage():
    def __init__(self, name:str, trainer:Trainer, num_epochs:int|None=None, train:str|list[str]|None=None):
        self.name = name
        self.trainer = trainer
        self.num_epochs = num_epochs

        # ensure list
        if isinstance(train, str) or train is None:
            self.train = [train] if train is not None else []
        else: 
            self.train = list(train)

    def apply(self, model:nn.Module):
        # freeze all params
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze specified modules
        for module_name in self.train:
            if not hasattr(model, module_name):
                raise AttributeError(f"[Stage {self.name}] model has no submodule '{module_name}'")
            
            module: nn.Module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = True

class MultiTrainer(Trainer):
    def __init__(self, stages: list[MultiTrainerStage]):
        self.stages = stages
        self._orig_kwargs = {s.name: getattr(s.trainer, "_orig_kwargs", type(s.trainer).__name__) for s in stages}

    def run(
        self,
        model: nn.Module,
        loader: Loader,
        num_epochs: int,
        report_metrics: str | list[str] | None = None,
        verbose: bool = False,
    ):
        # init outputs to match Trainer behavior
        self.dev_metrics = {}
        self.test_metrics = {}
        self.test_values = {}
        self.test_report = ''

        # carry-forward model across stages
        cur_model = model

        for stage in self.stages:
            # freeze/train modules as specified    
            stage.apply(cur_model)

            # run stage
            stage.trainer.run(
                model = cur_model,
                loader = loader,
                num_epochs = stage.num_epochs if stage.num_epochs is not None else num_epochs,
                report_metrics = report_metrics,
                verbose = verbose,
            )

            # aggregate dev_metrics
            for epoch, phases in stage.trainer.dev_metrics.items():
                self.dev_metrics.setdefault(epoch, {})
                for phase, metrics in phases.items(): # phase: 'train' or 'val'
                    self.dev_metrics[epoch].setdefault(phase, {})                    
                    self.dev_metrics[epoch][phase].update({f"{k}_{stage.name}": v for k,v in metrics.items()})

            # aggregate test_metrics, test_values, test_report
            tm = getattr(stage.trainer, "test_metrics", {}) or {}
            tv = getattr(stage.trainer, "test_values", {}) or {}

            for k,v in tm.items():
                self.test_metrics[f"{k}_{stage.name}"] = v

            for k,v in tv.items():
                self.test_values[f"{k}_{stage.name}"] = v

            # carry forward trained model
            cur_model = stage.trainer.model

        # final outputs
        self.model = cur_model

        self.test_report = ' | '.join(
            f'{stage.name}: {stage.trainer.test_report}'
            for stage in self.stages
            if getattr(stage.trainer, 'test_report', None)
        )


# Experiment
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
        # clean name
        name = clean_name(name)

        # generate unique name if duplicates
        base = name
        candidate = name
        i = 0
        while candidate in self.configs:
            candidate = f'{base}_{i}'
            i += 1

        # use updated name, with warning, if duplicate
        if candidate != name:
            print(f'Warning: {name} already in configs, adding as {candidate}')
            name = candidate
            
        # add to dicts
        self.models[name] = model
        self.configs[name] = trainer

    def add_grid(self, model_grid:nn.Module|dict[str,nn.Module], trainer_grid:Trainer|dict[str,Trainer], prefix:str|None=None, suffix:str|None=None):
        # convert to dict if instance provided
        if isinstance(model_grid, nn.Module):
            model_grid = {'':model_grid}
        if isinstance(trainer_grid, Trainer):
            trainer_grid = {'':trainer_grid}

        # iterate over grids
        for model_name, model in model_grid.items():
            for trainer_name, trainer in trainer_grid.items():
                
                # build name; fallback as 'config' if both == ''
                parts = []
                if model_name: 
                    parts.append(clean_name(model_name))
                if trainer_name: 
                    parts.append(clean_name(trainer_name))
                name = '_'.join(parts) if parts else 'config'

                # add prefix, suffix
                if prefix:
                    name = f'{clean_name(prefix)}_{name}'
                if suffix:
                    name = f'{name}_{clean_name(suffix)}'

                # add config
                self.add_config(name, model, trainer)

    def run_experiment(self, comment:str=None, report_metrics:str|list[str]|None=None, save_csv:bool=False, save_params:bool=False, save_model:bool=False, save_values:bool=False, verbose:bool=True, loader_kwargs:dict=None):
        loader_kwargs = {} if loader_kwargs is None else loader_kwargs
        if report_metrics is None:
            report_metrics = ['loss']
        elif isinstance(report_metrics, str):
            report_metrics = [report_metrics]

        # make folder
        self.folder = self._get_folder(comment) if True in (save_csv, save_params, save_model, save_values) else None
 
        # init trackers
        dev_metrics = {}
        test_metrics = {}
        test_values = {}
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
            test_values[trial] = {}
            seeds.append(trial_loader.seed)

            # for each config
            config_pbar = tqdm(self.configs.items(), leave=False)
            for i, (config_name, config) in enumerate(config_pbar):
                # update pbar
                report = f'Trial {trial+1}/{self.num_trials}, Config: {i+1}/{len(self.configs)} ({config_name})'
                config_pbar.set_postfix_str(report)
    
                # run config with trial loader
                config.run(model=self.models[config_name], loader=trial_loader, num_epochs=self.num_epochs, report_metrics=report_metrics)

                # add trial config to trackers
                dev_metrics[trial][config_name] = config.dev_metrics
                test_metrics[trial][config_name] = config.test_metrics
                test_values[trial][config_name] = config.test_values

                # create subfolder
                if save_params or save_model:
                    subfolder = self.folder / f'{config_name}'
                    subfolder.mkdir(parents=True, exist_ok=True)

                # save config params
                if save_params:
                    # get path
                    config_params_path = subfolder / f'{config_name}_params.json'

                    # write if doesnt exist (avoid duplicates)
                    if not config_params_path.exists():

                        # get config params, clean names recursively
                        config_params = {'model': config.model._orig_kwargs, 'trainer': config._orig_kwargs}
                        config_params = get_name_recursive(config_params)

                        with config_params_path.open('w') as f:
                            json.dump(config_params, f, indent=4, default=str)

                # save model
                if save_model:
                    state_dict = config.model.state_dict()
                    torch.save(state_dict, subfolder / f'{config_name}_trial_{trial}_model.pth')

                # print report
                if verbose:
                    tqdm.write(f'{report},\t {config.test_report}')
                
            # save metrics, params per trial
            self.dev_df = self._format_outputs(dev_metrics, 'dev', save_csv)
            self.test_df = self._format_outputs(test_metrics, 'test', save_csv)
            self.test_values = self._format_outputs(test_values, 'values', save_values)
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
    
    def _format_outputs(self, x:dict, method:Literal['dev','test','values'], save:bool=False):
        if method in ('dev', 'test'):

            # format rows
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
            if save:
                df.to_csv(self.folder / f'{method}.csv', index=False)

            return df
        
        elif method == 'values':
            out = {}

            # restructure dict
            for trial, configs in x.items():        
                for config, names in configs.items():
                    for name, value in names.items():

                        # initialize config dict
                        if config not in out:
                            out[config] = {}

                        # initialize value list
                        if name not in out[config]:
                            out[config][name] = []

                        # append value to list
                        out[config][name].append(value)

            # save
            if save:
                # stack arrays, flatten
                flat = {}
                for config, names in out.items():
                    for name, values in names.items():
                        flat[f'{config}/{name}'] = np.stack(values, axis=0)
                
                # save to .npz
                np.savez_compressed(self.folder/f'values.npz', **flat)

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
            summary_df = df.groupby(['config','metric'])['value'].agg(mean='mean', sd='std', ci=_get_ci).reset_index()

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
                'experiment': expt_params
            }

            # clean names
            params = get_name_recursive(params)

            # save
            with open(self.folder / f'params.json', 'w') as f:
                json.dump(params, f, indent=4, default=str)

def grid(obj, *, prefix:str|None=None, suffix:str|None=None, merge_keys:str|list[str]|None=None, **param_lists):
    objs = {} # init model dict
    keys = list(param_lists.keys()) # get keys (each param)
    
    # norm merge keys
    if merge_keys is None:
        merge_keys = []
    if not isinstance(merge_keys, list):
        merge_keys = [merge_keys]

    # unpack if partial !!!
    if isinstance(obj, functools.partial):
        base_callable = obj.func
        base_kwargs = dict(obj.keywords or {})
        base_args = obj.args or ()

    else:
        base_callable = obj
        base_kwargs = {}
        base_args = ()

    # filter kwargs (safe func/class.__init__)
    base_callable = filter_kwargs(base_callable) # originally: obj = filter_kwargs(obj) !!!

    # for each value combination
    for values in itertools.product(*param_lists.values()):
        # get params
        params = dict(zip(keys,values))

        # build name (new params only)
        parts = [] 
        for k,v in params.items():
            if isinstance(v, bool):
                v = 'T' if v else 'F' # bool as T/F
            if isinstance(v, dict):
                for vk, vv in v.items():
                    ck = clean_name(vk).replace('_','')
                    cv = clean_name(vv).replace('_','')
                    cv = cv[:1].upper() + cv[1:] # capitalize first char
                    parts.append(f'{ck}{cv}')
                continue

            ck = clean_name(k).replace('_','')
            cv = clean_name(v).replace('_','')
            cv = cv[:1].upper() + cv[1:] # capitalize first char
            parts.append(f'{ck}{cv}')
        name = '_'.join(parts)

        # add prefix, suffix
        if prefix:
            name = f'{clean_name(prefix)}_{name}'
        if suffix:
            name = f'{name}_{clean_name(suffix)}'

        # merge partial !!!
        for k in merge_keys:
            if isinstance(params.get(k), dict) and isinstance(base_kwargs.get(k), dict):
                params[k] = {**base_kwargs[k], **params[k]} # grid overrides partial
        merged_kwargs = {**base_kwargs, **params}

        # build config name & model
        objs[name] = base_callable(*base_args, **merged_kwargs) # originally obj(**params) !!!

    return objs

def kwarg_grid(**param_lists):
    # get keys
    keys = param_lists.keys()

    # init list
    param_grid = []

    # for each value combination
    for values in itertools.product(*param_lists.values()):
        # get params
        params = dict(zip(keys, values))
        
        # append to list 
        param_grid.append(params)

    return param_grid