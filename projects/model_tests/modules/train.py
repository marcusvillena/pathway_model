import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.metrics import accuracy_score

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

class DataModule():
    def __init__(self, X, y, generator, batch_size:int=16, val_size:int=0.15, test_size:int=0.15):
        # format Xy as dataset
        self.dataset = self.CustomDataset(X, y)

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

        # get class_weights
        self.class_weights = y.shape[0]/y.sum(dim=0)

    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

class MultiTrainingModule():
    def __init__(self, model_class, model_kwargs, training_class, training_kwargs, num_trials:int, comment = None):
        # get dir
        self.folder = self._get_dir(comment)

        # init trackers
        self.trial_metrics = {}
        self.test_metrics = {}

        # trial loop
        for trial in range(num_trials):

            # define model
            model = model_class(**model_kwargs)

            # define, run experiment
            training_module = training_class(model=model, **training_kwargs)

            # save trial results
            self.trial_metrics[trial] = training_module.trial_metrics
            self.train_results = self._trial_to_csv(self.trial_metrics, 'train')
            self.val_results = self._trial_to_csv(self.trial_metrics, 'val')

            # save test results
            self.test_metrics[trial] = training_module.test_metrics
            self.test_results = self._test_to_csv(self.test_metrics)

            # save model
            state_dict = model.state_dict()
            torch.save(state_dict, self.folder / f'model_trial_{trial}.pth')

    def _get_dir(self, comment):
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

    def _trial_to_csv(self, metrics_dict, col:str):
        # trial dict to dataframe
        results = pd.DataFrame(
            [
                {'trial':trial, 'epoch':epoch, **epoch_metrics[col]} # col: 'train' or 'val'
                for trial, trial_metrics in metrics_dict.items() # 
                for epoch, epoch_metrics in trial_metrics.items()
            ]
        )
        
        # write to csv
        results.to_csv(self.folder / f'{col}.csv')

        return results

    def _test_to_csv(self, metrics_dict, col:str='test'):
        # test dict to dataframe
        results = pd.DataFrame(
            [
                {'trial':trial, **metrics}
                for trial, metrics in metrics_dict.items()
            ]
        )

        # write to csv
        results.to_csv(self.folder / f'{col}.csv')

        return results

class TrainingModule():
    def __init__(self, model, data_module, loss_fn, optimizer, num_epochs:int, report_metrics=['tot_loss'], optimizer_kwargs={}):
        # assign variables
        self.model = model
        self.data_module = data_module
        self.loss_fn = loss_fn
        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)

        # train val loop
        self.trial_metrics = {}
        pbar = tqdm(range(num_epochs))
 
        for epoch in pbar:
            # training
            train_metrics =  self._train_model(self.data_module.train_loader)

            # validation
            val_metrics = self._eval_model(self.data_module.val_loader)

            # record training/validation
            self.trial_metrics[epoch] = {'train': train_metrics, 'val': val_metrics}
            
            # get reports
            train_report = self._generate_report(train_metrics, report_metrics)
            val_report = self._generate_report(val_metrics, report_metrics)

            # update pbar with report
            epoch_report = f'Epoch {epoch:<8}' + f'Train: {train_report}' + 8*' ' + f'Val: {val_report}'
            pbar.set_postfix_str(epoch_report)

        # test
        self.test_metrics = self._eval_model(self.data_module.test_loader)

        # print test report
        test_report = self._generate_report(self.test_metrics, report_metrics)
        tqdm.write(f'Test\t {test_report}\n')

    def _compute_loss(self, X, y): # change in child obj if needed
        # get model output
        out = self.model(X)

        # compute loss
        loss = self.loss_fn(out, y)

        return loss, out

    def _get_y_pred(self, X): # change in child obj if needed
        # get prediction
        y_pred = self.model.predict(X)

        return y_pred

    def _compute_metrics(self, batch): # change in child obj if needed
        # init
        metrics = {}

        # compute metrics
        metrics['tot_loss'] = batch['tot_loss']/batch['len']

        return metrics
    
    def _generate_report(self, metrics:dict, report_metrics:list):
        # generate report
        report = (4*' ').join(
            f'{metric}={metrics[metric]:<.4f}'
            for metric in report_metrics
            if metric in metrics
        )

        return report

    def _train_model(self, dataloader):
        # set model to train
        self.model.train()

        # init performance trackers
        batch = {'batch':0, 'len':len(dataloader), 'tot_loss':0, 'y':[], 'y_pred':[]}

        # iterate over data
        for X, y in dataloader:
            
            self.optimizer.zero_grad()

            # compute loss, pred
            loss, out = self._compute_loss(X, y)
            y_pred = self._get_y_pred(out)
            
            # backprop
            loss.backward()
            self.optimizer.step()

            # update performance
            batch = self._update_performance(batch, loss, y, y_pred)

        # get performance metrics
        metrics = self._compute_metrics(batch)

        return metrics
    
    def _eval_model(self, dataloader):
        # set model to eval
        self.model.eval()

        # init performance trackers
        batch = {'batch':0, 'len':len(dataloader), 'tot_loss':0, 'y':[], 'y_pred':[]}

        # iterate over data
        with torch.no_grad():
            for X, y in dataloader:
                
                # compute loss, pred
                loss, out = self._compute_loss(X, y)
                y_pred = self._get_y_pred(out)

                # update performance
                batch = self._update_performance(batch, loss, y, y_pred)

        # get performance metrics
        metrics = self._compute_metrics(batch)

        return metrics
    
    def _update_performance(self, batch, loss, y, y_pred):
        # increment batch, loss
        batch['batch'] += 1
        batch['tot_loss'] += loss.item()

        # update preds, ys for other metrics
        batch['y'].extend(y.detach().cpu().numpy())
        batch['y_pred'].extend(y_pred.detach().cpu().numpy())

        return batch

class ClassifierTrainingModule(TrainingModule):
    def _compute_metrics(self, batch): # change in child obj if needed
        # init
        metrics = {}

        # compute metrics
        metrics['tot_loss'] = batch['tot_loss']/batch['len']
        metrics['accuracy'] = accuracy_score(batch['y'], batch['y_pred'])

        return metrics
        
class Experiment():
    def __init__(
            self, data, generator, model_class:nn.Module, training_class:TrainingModule, # required
            batch_size:int=64, val_size:int=0.15, test_size:int=0.15, # for data_module
            model_kwargs:dict={}, training_kwargs:dict={}, num_trials:int=10, comment:str=None # for training_module
        ):
        
        # define data module
        self.data_module = DataModule(
            X=data.X,
            y=data.y,
            generator=generator,
            batch_size=batch_size,
            val_size=val_size,
            test_size=test_size
        )

        # run experiment
        self.training_module = MultiTrainingModule(
                model_class=model_class,
                model_kwargs=model_kwargs,
                training_class=training_class,
                training_kwargs={
                    'data_module': self.data_module,
                    **training_kwargs
                },
                num_trials=num_trials,
                comment=comment
        )

        # assign instance vars
        self.comment = comment

    def results(self, metric:str, plot:bool=True, title:bool=True):
        # copy data
        train = self.training_module.train_results.copy()
        val = self.training_module.val_results.copy()
        test = self.training_module.test_results.copy()

        # add stage, get combined df
        train['stage'] = 'Training'
        val['stage'] = 'Validation'
        dev = pd.concat([train, val])

        # get test metrics
        test_mean = test[metric].mean()
        test_std = test[metric].std()

        # return metrics if no plot
        if plot != True:
            return test_mean, test_std
        
        # return metrics with plot
        else:
            # define plot
            fig, ax = plt.subplots(figsize=(16, 9))

            # plot dev set results
            sns.lineplot(data=dev, x='epoch', y=metric, hue='stage', errorbar='sd')

            # plot test set line
            ax.hlines(
                y=test_mean, 
                xmin=min(dev['epoch']),
                xmax=max(dev['epoch']),
                colors='green', 
                linestyles='--', 
                label='Testing (trial mean ± SD)'
            )

            # plot test set area
            ax.fill_between(
                x=dev['epoch'].unique(),
                y1=test_mean - test_std,
                y2=test_mean + test_std,
                color='green',
                alpha=0.2,
            )

            # add test results label
            ax.text(
                x=max(dev['epoch']), # x pos = final (max) epoch
                y=test_mean + 2*test_std, # y pos = 2 std above mean
                s=f'{test_mean:.4f} ± {test_std:.4f}', # label
                va='bottom', # vert align
                ha='right', # horz align
                color='green',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='green', alpha=0.7)
            )

            # lazy title
            if title == True:
                plt.title(f'{self.comment}: {metric.capitalize()}')
            
            # axis labels
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend(title='Stage')
            ax.grid(True)
            fig.tight_layout()

            # return test results
            return test_mean, test_std, fig






        

