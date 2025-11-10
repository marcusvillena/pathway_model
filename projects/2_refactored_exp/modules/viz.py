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

from torch_geometric.loader import DataLoader
from typing import Literal

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