import torch
import torch.nn as nn
import torch.optim as optim

from .data import GraphDataset
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
    def __init__(self, model, loader:Loader, num_epochs:int, loss_fn:nn.Module, optimizer_class:optim.Optimizer=optim.Adam, optimizer_kwargs:dict={}, report_metrics=['loss'], verbose:bool=False, autorun:bool=True):
        # assign inst vars
        self.model = model # should be a predefined model
        self.loader = loader
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        self.report_metrics = report_metrics
        self.verbose = verbose
        
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
            train_metrics = self._run_phase('train', self.loader.train_loader)

            # validating
            val_metrics = self._run_phase('eval', self.loader.val_loader)

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
        self.test_metrics = self._run_phase('eval', self.loader.test_loader)

        # print test report
        if self.verbose == True:
            test_report = self._generate_report(self.test_metrics, self.report_metrics)
            tqdm.write(f'Test\t {test_report}\n')

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

            self.batch_log = batch_log # debugging

        # eval mode
        else:
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    loss, out = self._compute_loss(batch)
                    batch_log = self._update_batch_log(batch_log, loss, batch, out)

            # self.batch_log = batch_log # debugging

        # compute metrics
        metrics = self._compute_metrics(batch_log)

        return metrics
    
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
            return item.detach().cpu()

        # list/tuple of tensors
        elif isinstance(item, (tuple, list)):
            return type(item)(self._detach_items(i) for i in item)

        # dict
        elif isinstance(item, dict):
            return {key: self._detach_items(value) for key, value in item.items()}
        
        # PyG DataBatch or other class with .x
        elif hasattr(item, 'x'):
            x = self._detach_items(item.x)
            y = self._detach_items(getattr(item, 'y', None))
            return {'x': x, 'y': y} if y is not None else {'x':x}
        
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

        # compute metrics
        metrics['loss'] = batch_log['loss']/batch_log['num_batches']

        return metrics