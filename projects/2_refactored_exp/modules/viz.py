from .utils import clean_name, validate_filter, filter_df
from pathlib import Path
from statannotations.Annotator import Annotator
import itertools
import json
from scipy import stats
import shutil

##

import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors

import statsmodels.api as sm

import umap
# import sklearn.metrics
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# typing
from pandas import DataFrame
from torch_geometric.loader import DataLoader
from typing import Literal


# summary stats
def test_summary(df:pd.DataFrame, save_csv:bool=False, filename:str|Path='summary.csv'):

    if isinstance(filename, str):
        filename = Path(filename)

    # mean, std, ci if trials > 1
    if df['trial'].max() > 0:
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
        summary_df.to_csv(filename, index=False)

    return summary_df

class ConfigLookup(): # requires json
    def __init__(self, *, keys:str|list[str]|None=None, path:str|Path|None=None, **kwargs):
        if keys is None:
            return
        
        if path is not None:
            self.from_expt_folder(keys=keys, path=path, **kwargs)
            return 
        
        if 'filepaths' in kwargs:
            filepaths = kwargs["filepaths"]
            use_keypath = kwargs.get("use_keypath", False)
            self.from_filepaths(keys=keys, filepaths=filepaths, use_keypath=use_keypath)
        
    def from_expt_folder(self, keys:list[str], path:str|Path, configs:str|list[str]|None=None, use_keypath:bool=False, save:bool=False):
        path = Path(path)
        
        # config fallback: use all from test.csv
        if configs is None:
            configs = pd.read_csv(path/'test.csv')['config'].unique().tolist()
        configs: list[str] = configs if isinstance(configs, list) else [configs]

        # get filepaths
        filepaths: list[Path] = [path / config / f'{config}_params.json' for config in configs]

        # run lookup
        self.from_filepaths(keys, filepaths, use_keypath=use_keypath)

        # save
        name = '_'.join([clean_name(i) for i in self.keys])
        self.data.to_csv(path / f'{name}_conf.csv', index=False)

    def from_filepaths(self, keys: str | list[str], filepaths: str | Path | list[str|Path], use_keypath:bool=False):
        # ensure types
        self.keys = keys if isinstance(keys, list) else [keys]
        self.filepaths = filepaths if isinstance(filepaths, list) else [filepaths]
        self.filepaths:list[Path] = [Path(i) for i in self.filepaths]

        if not self.filepaths:
            raise ValueError("No filepaths provided to ConfigLookup.")

        # get keypaths from first config
        with open(self.filepaths[0]) as f:
            d = json.load(f)
        self.keypaths = self._find_keypaths(self.keys, d)

        # get values using keypaths
        self.data = []
        for filepath in self.filepaths:
            with open(filepath) as f:
                j = json.load(f)
            d = self._path_to_dict(self.keypaths, j, use_keypath)

            # get config name
            stem = filepath.stem
            d["config"] = stem[:-len('_params')] if stem.endswith('_params') else stem
            self.data.append(d)

        # convert to dataframe
        self.data = pd.DataFrame(self.data)

    def _flatten_dict(self, d:dict, parent=()):
        out = {}

        if isinstance(d, dict):
            iterator = d.items()
        elif isinstance(d, list):
            out[parent] = d
            iterator = enumerate(d)
        else:
            out[parent] = d
            return out

        for k, v in iterator:
            path = parent + (k,)
            if isinstance(v, (dict, list)):
                out.update(self._flatten_dict(v, path))
            else:
                out[path] = v

        return out

    def _find_keypaths(self, find, d):
        if not isinstance(find, list):
            find = [find]

        flat = self._flatten_dict(d)

        # index by final key
        index = {}
        for path in flat.keys():
            last = path[-1]
            index.setdefault(last, []).append(path)

        # flatten the result (single list of tuples)
        result = []
        for name in find:
            result.extend(index.get(name, []))

        return result

    def _path_to_dict(self, keypaths, d, use_keypath:bool=False):
        out = {}

        for keypath in keypaths:
            
            # start at d
            current = d

            # traverse path
            for key in keypath:
                current = current[key]

            # append final key with value at end of path
            if use_keypath:
                out[keypath] = current
            else:
                out[key] = current

        return out

class MultiExperiment():
    def __init__(
        self,
        experiment_dirs: str | Path | list[str|Path],
        keys: str | list[str],
        out_dir: str | Path = './experiments',
        copy_files: bool = False,
        use_keypath: bool = False,
        overwrite: bool = False
    ):
        self.keys = keys if isinstance(keys, list) else [keys]
        self.out_dir = Path(out_dir)
        self.overwrite = overwrite
        self.avoided_overwrite: list = []

        # make output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ensure list and paths are children of parent
        if not isinstance(experiment_dirs, list):
            experiment_dirs = [experiment_dirs]
        self.experiment_dirs = [Path(exp_dir) for exp_dir in experiment_dirs]

        # merge devtest csvs
        self.dev: DataFrame = self._merge_devtests('dev.csv')
        self.test: DataFrame = self._merge_devtests('test.csv')
        self.summary: DataFrame = test_summary(self.test)

        # copy config files
        self.config_names: list[str] = self.test['config'].unique().tolist()
        self._copy_config_files()

        # copy experiment files
        if copy_files:
            self._copy_expt_files()

        # get configs
        self.config_lookup = ConfigLookup(
            keys=self.keys,
            filepaths=[self.config_dir / f'{name}_params.json' for name in self.config_names],
            use_keypath=use_keypath,
        )
        self.configs = self.config_lookup.data

        # save csvs
        self.dev = self._save_csv(self.dev, 'dev.csv')
        self.test = self._save_csv(self.test, 'test.csv')
        self.summary = self._save_csv(self.summary, 'summary.csv')
        self.configs = self._save_csv(self.configs, 'configs.csv')

        # print overwrite msg
        if self.avoided_overwrite:
            print(f'Avoided overwriting {len(self.avoided_overwrite)} files. Call self.avoided_overwrite to see list, or run with overwrite=True to overwrite.')
            
    def _merge_devtests(self, file: str) -> DataFrame:
        expt_dfs: list[DataFrame] = []
        old_max = -1 # start at -1 + 1 = 0

        for expt_dir in self.experiment_dirs:
            df = pd.read_csv(expt_dir/file)

            # align trials
            new_min =  df['trial'].min() # min of current expt
            shift = (old_max + 1) - new_min # shift to align with old max + 1
            df['trial'] += shift # shift trials to start after old max
            old_max = df['trial'].max() # update old max for next expt

            expt_dfs.append(df)

        return pd.concat(expt_dfs, ignore_index=True)
    
    def _copy_fn(self, src: str, dst: str) -> str:
        if not Path(dst).exists() or self.overwrite:
            return shutil.copy2(src, dst)
        else:
            self.avoided_overwrite.append(str(dst))
            return dst
        
    def _copy_config_files(self):
        # make config directory
        self.config_dir = self.out_dir / 'configs'
        self.config_dir.mkdir(parents=True, exist_ok=True)    

        # copy config _params.json files, once per config
        for config in self.config_names:
            for expt in self.experiment_dirs:
                src = expt / config / f'{config}_params.json'
                if src.exists(): # if config in expt
                    dst = self.config_dir / f'{config}_params.json'
                    self._copy_fn(src, dst)
                    break

    def _copy_expt_files(self):
        # make file directory
        self.file_dir = self.out_dir / 'files'
        self.file_dir.mkdir(parents=True, exist_ok=True)

        # copy expt files into file directory
        for expt in self.experiment_dirs:
            expt = expt.resolve()
            if not expt.is_dir():
                continue # skip non-directories

            shutil.copytree(
                src=str(expt),
                dst=str(self.file_dir / expt.name),
                dirs_exist_ok=True,
                copy_function=self._copy_fn
            )

    def _save_csv(self, df: DataFrame, filename: str,  merge:bool=True):
        if merge and filename != 'configs.csv':
            df = pd.merge(self.configs, df, on='config')

        out_path = self.out_dir / filename
        
        if not out_path.exists() or self.overwrite:
            df.to_csv(out_path, index=False)
        else:
            self.avoided_overwrite.append(str(out_path))

        return df

# grid experiment plots
def metric_x_point(
    # data
    df:pd.DataFrame, cols:list[str], metrics:list[str]|None=None, filters:dict|None=None, 
    
    # plot
    hue:bool=False, strip:bool=False, alpha:float=0.5, figsize:tuple|None=None, dodge:bool=True, 

    # stats
    sig:Literal['within','between']|bool=True, test:str='t-test_ind'
):
    # defaults
    if filters is None: filters = {}
    metrics = validate_filter(metrics, df['metric'].unique(), 'metrics')
    if not isinstance(cols, list): cols = [cols]

    # define plotting func
    def mxpointplot(data, x:str, metric:str, hue_col=None):
        
        # get dodge amounts
        if dodge and hue:
            dodge_point = 0.8 - 0.8 / len(data[hue_col].unique())
        else:
            dodge_point = False

        # build plots
        fig, ax = plt.subplots(figsize=figsize)
        if strip:
            sns.stripplot(data=data,x=x, y='value', hue=hue_col, ax=ax, alpha=alpha, legend=False, dodge=dodge) #dodge=True)
        sns.pointplot(data=data, x=x, y='value', hue=hue_col, ax=ax, dodge=dodge_point)
        ax.set_xlabel(x.capitalize())
        ax.set_ylabel(metric.capitalize())
        plt.tight_layout()

        # no sig
        if not sig:
            return ax

        # sig, no hue
        elif hue_col is None:
            # get pairs
            levels = sorted(list(data[x].unique()))
            pairs = list(itertools.combinations(levels,2))

            # skip if no pairs (empty)
            if not pairs:
                return ax

            # build annotations if pairs
            annot = Annotator(ax, pairs, data=data, x=x, y='value')
  
        # sig, hue
        else:
            # build pairs if none
            x_levels = sorted(data[x].dropna().unique())
            h_levels = sorted(data[hue_col].dropna().unique())
            pairs = []
            
            if sig == 'within' or sig == True:
                pairs = [
                    ((xv,h1), (xv,h2))
                    for xv in x_levels
                    for (h1,h2) in itertools.combinations(h_levels,2)
                ]
            elif sig == 'between':
                pairs = [
                    ((x1,hv), (x2,hv))
                    for hv in h_levels
                    for (x1,x2) in itertools.combinations(x_levels,2)
                ]
            else:
                raise ValueError("sig_mode must be 'within' or 'between'")

            # skip if no pairs (empty)
            if not pairs:
                return ax
            
            # build annotations if pairs
            annot = Annotator(ax, pairs, data=data, x=x, y='value', hue=hue_col)

        # add stars, apply annotator
        annot.configure(test=test, text_format='star', loc='inside', verbose=False, hide_non_significant=True)
        annot.apply_and_annotate()

        return ax

    # plotting loop
    for metric in metrics:

        # add metric to filter
        filters['metric'] = metric

        # filter for metric
        filt = filter_df(df, filters)

        # plot col1(x) * col2(hue), per col, per metric
        if hue:
            perms = list(itertools.permutations(cols, 2))
            for perm in perms:
                mxpointplot(data=filt, x=perm[0], metric=metric, hue_col=perm[1])

        # plot across all, per col, per metric
        else:
            for col in cols:
                mxpointplot(data=filt, x=col, metric=metric, hue_col=None)

# Expt
def devplot(
    dev:pd.DataFrame,
    summary:pd.DataFrame|None=None,
    configs:list[str]|None=None, 
    metrics:list[str]|None=None,
    figsize:tuple[int]|None=None,
    errorbar:Literal['ci','sd']|None='ci',
    save_folder:str|None=None,
):
    # default: plot all
    configs = validate_filter(configs, dev['config'].unique(), 'configs')
    metrics = validate_filter(metrics, dev['metric'].unique(), 'metrics')

    # rename stage for legend title
    dev['stage'] = dev['stage'].replace({'train':'Training', 'val':'Validation'})

    # store figs in dict
    figs = {}

    # for each config & metric pair
    for config in configs:

        # subdict per config
        figs[config] = {}

        for metric in metrics:

            # filter dev df
            dev_filt = dev[(dev['config']==config) & (dev['metric']==metric)]

            # plot dev
            plt.figure(figsize=figsize)
            sns.lineplot(data=dev_filt, x='epoch', y='value', hue='stage', errorbar=errorbar)

            # plot test
            if isinstance(summary, pd.DataFrame):
            
                # filter summary df
                summary_filt = summary[(summary['config']==config) & (summary['metric']==metric)]

                # plot mean
                test_mean = summary_filt['mean'].item()
                plt.axhline(
                    y=test_mean,
                    color='green',
                    linestyle='--',
                    label='Test'
                )
                test_label = f'{test_mean:.4f}' # make label

                # plot err
                if isinstance(errorbar, str) and (errorbar in summary.columns):
                    test_err = summary_filt[errorbar].item()
                    plt.fill_between(
                        x = dev_filt['epoch'],
                        y1 = test_mean - test_err,
                        y2 = test_mean + test_err,
                        color='green',
                        alpha=0.2,
                    )
                    test_label = f'{test_label} ± {test_err:.4f}' # add err to label

                # add test label
                plt.text(
                    x = max(dev_filt['epoch']),
                    y = test_mean,
                    s = test_label,
                    va = 'center',
                    ha = 'left',
                    color = 'green',
                    bbox = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='green', alpha=0.9)
                )

            # formatting
            plt.title(f'{config.capitalize()} | {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend(title='Stage')
            plt.tight_layout() 

            # write to dict, save, close
            fig = plt.gcf()
            figs[config][metric] = fig
            if save_folder is not None:
                path = Path(f'{save_folder}/{config}/')
                path.mkdir(parents=True, exist_ok=True)
                fig.savefig(path/f'{config}_{metric}_devplot.svg')
            plt.close()

    return figs

def testplot(test:pd.DataFrame, configs:list=None, metrics:list=None):
    # set defaults
    configs = configs if isinstance(configs, list) else test['config'].unique()
    metrics = metrics if isinstance(metrics, list) else test['metric'].unique()

    # get n_configs (for figsize)
    n_configs = len(configs)

    # plot per metric
    for metric in metrics:

        # filter df for metric and configs
        metric_df = test[test['config'].isin(configs)]
        metric_df = metric_df[metric_df['metric'] == metric]

        plt.figure(figsize=(8, 0.5*n_configs))
        ax = sns.pointplot(
            data=metric_df, 
            y='config', 
            x='value', 
            errorbar=('ci',95), 
            linestyle='none'
        )

        ax.set_xlabel(metric.capitalize())
        ax.set_ylabel('Configuration')

## Single model viz

class ModelOutput():
    def __init__(self, model, dataset, batch_size:int=64):
        self.values = {}

        # initialize loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # run model
        out = {}
        x = {}
        y = {}
        batch_num = 0
        with torch.inference_mode():
            for batch in loader:
                x[batch_num] = batch.x
                y[batch_num] = batch.y
                out[batch_num] = model.get_weights(batch)
                batch_num += 1 

        # combine batches into lists
        for batch, outputs in out.items():
            for k,v in outputs.items():
                self.values.setdefault(k,[]).append(v)
        
        # combine list of tensors
        def is_tensor_list(lst):
            return isinstance(lst, list) and all(isinstance(x, torch.Tensor) for x in lst)

        self.values = {
            k: torch.cat(v, dim=0).detach().cpu().numpy() if is_tensor_list(v) else v
            for k,v in self.values.items()
        }

        # add xy
        self.values['x'] = torch.cat([i for i in x.values()], dim=0).detach().cpu().numpy()
        self.values['y'] = torch.cat([i for i in y.values()], dim=0).detach().cpu().numpy()

def x_recon_heatmap(
    model_output, 
    num_nodes:int, 
    class_names, 
    x_in:str, 
    x_out:str, 

    in_title:str=None, 
    out_title:str=None, 
    cbar_title:str=None, 
    row_whitespace:int = 1, 
    transform:Literal['log','exp']=None, 
    borders:bool=True, 
    figsize:tuple[int]=(14, 5)
):
        # --- Input data ---
        x_in = model_output.values[f'{x_in}'].reshape(-1, num_nodes)
        x_out = model_output.values[f'{x_out}'].reshape(-1, num_nodes)
        labels = model_output.values['y']

        if transform == 'log':
            x_in = np.log(x_in+1)
            x_out = np.log(x_out+1)
        elif transform == 'exp':
            x_in = np.exp(x_in)
            x_out = np.exp(x_out)

        in_title = in_title if in_title is not None else 'Data'
        out_title = out_title if out_title is not None else 'Model prediction'

        # --- Class metadata ---
        assert max(labels) < len(class_names), "Label index exceeds class_names"

        # --- Column clustering ---
        col_order = sch.dendrogram(sch.linkage(x_in.T, method='average'), no_plot=True)['leaves']
        x_in = x_in[:, col_order]
        x_out = x_out[:, col_order]

        # --- Cluster rows within each class ---
        unique_classes = np.unique(labels)
        row_order = []
        x_in_blocks, x_out_blocks = [], []
        block_sizes = []

        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            cls_x = x_in[cls_indices]

            if cls_x.shape[0] < 2:
                cls_order = cls_indices
            else:
                row_linkage = sch.linkage(cls_x, method='average')
                cls_order = cls_indices[sch.dendrogram(row_linkage, no_plot=True)['leaves']]

            x_in_blocks.append(x_in[cls_order])
            x_out_blocks.append(x_out[cls_order])
            block_sizes.append(len(cls_order))
            row_order.extend(cls_order.tolist())

        # --- Add white space between class blocks ---
        white_row = np.full((row_whitespace, x_in.shape[1]), np.nan)

        x_in_padded, x_out_padded = [], []
        label_ticks = []
        curr_idx = 0

        for i, (xin_block, xout_block, size) in enumerate(zip(x_in_blocks, x_out_blocks, block_sizes)):
            x_in_padded.append(xin_block)
            x_out_padded.append(xout_block)
            label_ticks.append(curr_idx + size // 2)
            curr_idx += size

            if i < len(x_in_blocks) - 1:
                x_in_padded.append(white_row)
                x_out_padded.append(white_row)
                curr_idx += row_whitespace

        # --- Stack padded arrays ---
        x_in_final = np.vstack(x_in_padded)
        x_out_final = np.vstack(x_out_padded)

        # --- Plotting ---
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        vmin = 0
        vmax = np.percentile(np.nan_to_num(x_in_final), 80)

        # Get actual gene index labels from col_order (before reordering)
        start_label = '0'
        end_label = str(num_nodes - 1)
        midpoint = x_in_final.shape[1] // 2

        # --- Heatmap: Data ---
        sns.heatmap(x_in_final, ax=ax0, vmin=vmin, vmax=vmax, cmap="rocket",
                    cbar=False, mask=np.isnan(x_in_final))
        ax0.set_title(in_title)
        ax0.set_yticklabels([])
        ax0.set_xticks([0, midpoint, x_in_final.shape[1]-1])
        ax0.set_xticklabels([start_label, 'Genes', end_label])
        ax0.tick_params(axis='x', rotation=0)

        # --- Heatmap: Model prediction ---
        sns.heatmap(x_out_final, ax=ax1, vmin=vmin, vmax=vmax, cmap="rocket",
                    cbar=False, mask=np.isnan(x_out_final))
        ax1.set_title(out_title)
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_xticks([0, midpoint, x_out_final.shape[1]-1])
        ax1.set_xticklabels([start_label, 'Genes', end_label])
        ax1.tick_params(axis='x', rotation=0)

        # --- Y-axis class names ---
        ax0.set_yticks(label_ticks)
        ax0.set_yticklabels([class_names[i] for i in unique_classes], rotation=0)

        # --- Shared colorbar ---
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap="rocket", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        if cbar_title is not None:
            cbar.set_label(cbar_title, labelpad=15)

        # --- Borders ---
        if borders:
            # # Add black rectangles around each class block
            y = 0
            for size in block_sizes:
                for ax in [ax0, ax1]:
                    rect = Rectangle((0, y), x_in.shape[1], size, linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)
                y += size + row_whitespace

        plt.tight_layout()
        plt.show()

def x_recon_scatter(
    model_output, 
    num_nodes:int, 
    # class_names, 
    x_in:str, 
    x_out:str, 
    
    in_title:str=None, 
    out_title:str=None, 
    transform:Literal['log','exp']=None
):
    
    # --- Input data ---
    x_in = model_output.values[f'{x_in}'].reshape(-1, num_nodes)
    x_out = model_output.values[f'{x_out}'].reshape(-1, num_nodes)
    labels = model_output.values['y']

    if transform == 'log':
        x_in = np.log(x_in+1)
        x_out = np.log(x_out+1)
    elif transform == 'exp':
        x_in = np.exp(x_in)
        x_out = np.exp(x_out)

    x = x_in
    x_recon = x_out
    x_mean = x_in.mean(axis=0)
    x_recon_mean = x_out.mean(axis=0)

    in_title = 'Mean value (per gene)' if in_title is None else in_title
    out_title = 'Reconstructed mean value (per gene)' if out_title is None else out_title

    #### plot ####
    
    # get MAE
    recon_error = np.array([mean_absolute_error(x[:, i], x_recon[:, i]) for i in range(x.shape[1])])
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
    plt.xlabel(in_title)
    plt.ylabel(out_title)
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

def embedding_scatter(
    model_output, 
    num_samples:int, 
    x:str,
    class_labels:list,   # class_labels[i] = name for class id i
):
    X = model_output.values[f'{x}'].reshape(num_samples, -1)
    y = np.asarray(model_output.values['y'])

    # sanity
    if y is None:
        raise ValueError("model_output.values['y'] must be provided.")
    unique_classes = np.unique(y)
    K = int(unique_classes.max()) + 1  # assumes classes are 0..K-1

    # Discrete colormap & norm so colors are per-class
    cmap = ListedColormap([plt.get_cmap('tab10')(i) for i in range(max(K, 10))][:K])
    norm = BoundaryNorm(boundaries=np.arange(-0.5, K+0.5, 1), ncolors=K)

    # Embeddings
    X_pca  = PCA(n_components=2).fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
    X_umap = umap.UMAP(n_components=2, random_state=None, n_jobs=-1).fit_transform(X)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ['PCA', 't-SNE', 'UMAP']
    results = [X_pca, X_tsne, X_umap]

    for ax, result, name in zip(axes, results, methods):
        ax.scatter(
            result[:, 0], result[:, 1],
            c=y, s=10, alpha=0.8,
            cmap=cmap, norm=norm, edgecolors='none'
        )
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    # --- Build a single shared legend (aligned with last subplot) ---
    legend_classes = unique_classes.tolist()
    legend_handles = [Patch(facecolor=cmap(norm(c)), edgecolor='none') for c in legend_classes]
    legend_labels  = [class_labels[int(c)] if int(c) < len(class_labels) else str(c) 
                      for c in legend_classes]

    # Place legend relative to last subplot
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Class",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),   # right edge of last axis
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.show()