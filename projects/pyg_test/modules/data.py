import pandas as pd
import numpy as np
import torch

from pathlib import Path
from .utils import vprint

def print_metadata(tcga_project:str, tcga_dir:str, return_df:bool=False):
    # read df
    df = pd.read_csv(Path(tcga_dir) / f'{tcga_project}_metadata.csv')

    # drop 'Unnamed: 0' col if applicable
    if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')

    # for each metadata column
    for i in df.columns:
        # get unique items per column
        unique = df[i].unique().tolist()

        # print items per column
        if len(unique) > 15:
            print(f'{i} ({len(unique)}): {unique[0:3]+['...']}\n') # shorten if > 15 unique
        else:
            print(f'{i} ({len(unique)}): {unique}\n')

    if return_df == True:
        return df

class TCGAData():
    def __init__(
            self,
            # dataset 
            tcga_project:str,
            tcga_dir:str,
            relation_filepath:str,

            # metadata
            metadata_subtype_col:str='',
            metadata_ctrl:str='Normal',

            # gene_counts
            log0_method:str='log1p',
            scale_gene_counts:bool=True, 
            fc_gene_counts:bool=True, 
            log2fc_gene_counts:bool=True,

            # filter, resample, etc.
            y_col:str='type',
            drop:list[str]=None,
            max_subset:int=None,
            verbose:bool=True,
        ):
        
        vprint('**** Data() ****', verbose=verbose)

        # assign
        self.log0_method = log0_method
        
        # read files into df
        gene_counts = self._read_gene_counts(tcga_project, tcga_dir)
        metadata = self._read_metadata(tcga_project, tcga_dir, metadata_subtype_col)
        relation = pd.read_csv(relation_filepath)

        # scale gene_counts
        if scale_gene_counts == True:
            gene_counts = self._scale_gene_counts(gene_counts)

        if (fc_gene_counts == True) & (metadata_ctrl != ''):
            gene_counts = self._fc_gene_counts(gene_counts, metadata, metadata_ctrl)

            if log2fc_gene_counts == True:
                gene_counts = np.log2(self._handle_log0(gene_counts))

        # preprocess gene_counts and relation into graph data
        gene_counts, relation, node_id_map = self._graph_preprocessing(gene_counts, relation)

        # get masks, flatten relation
        masks = self._get_masks(relation)
        relation = relation.drop(columns='pathway_name').groupby(['idx1','idx2'], as_index=False).any()

        # filter counts by class (drop classes, downsample) if applicable
        gene_counts, metadata = self._filter_counts(gene_counts, metadata, y_col, drop, max_subset)

        # get xy
        X, y, y_labels = self._get_Xy(gene_counts, metadata, y_col)

        # get dims
        num_samples, num_nodes, num_features, num_labels, num_masks = self._get_dims(X, y, masks)

        # get class_weights
        self.class_weights = y.shape[0]/y.sum(dim=0)
            
        # assign
        self.gene_counts = gene_counts
        self.metadata = metadata
        self.relation = relation
        self.node_id_map = node_id_map
        self.masks = masks
        self.X = X
        self.y = y
        self.y_labels = y_labels

        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_labels = num_labels
        self.num_masks = num_masks

        # print
        vprint(self, verbose=verbose)
        
    def _read_metadata(self, tcga_project:str, tcga_dir:str, metadata_subtype_col:str):
        # read df
        df_complete = pd.read_csv(Path(tcga_dir) / f'{tcga_project}_metadata.csv')

        # drop 'Unnamed: 0' col if applicable
        if 'Unnamed: 0' in df_complete.columns:
                df_complete = df_complete.drop(columns='Unnamed: 0')

        # compile df
        df = pd.DataFrame(
            {
                'barcode': df_complete['barcode'],
                'type': df_complete.apply(lambda row: row['name'] if row['tissue_type'] == 'Tumor' else row['tissue_type'], axis=1),
            }
        )

        # append subtype if applicable
        if metadata_subtype_col != '':
            df['subtype'] = df_complete[metadata_subtype_col].fillna(df_complete['sample_type'])

        return df

    def _read_gene_counts(self, tcga_project:str, tcga_dir:str):
        # read df
        df = pd.read_csv(Path(tcga_dir) / f'{tcga_project}_gene_counts.csv')

        # rename 'Unnamed: 0' col if applicable
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0':'ensg'})

        # remove ensg version
        df['ensg'] = df['ensg'].str.split('.').str[0]

        # set index, column name
        df = df.set_index('ensg')
        df.columns.name = 'barcode'

        return df
    
    def _handle_log0(self, x):
        # offset (replace 0 with small number)
        if self.log0_method == 'offset':
            if type(x) == pd.DataFrame:
                x = x.replace(0, 1e-6)
            else:
                x = x + 1e-6 # if x is not a df, add 1e-6

        # log1p method (add 1 before log)
        elif self.log0_method == 'log1p':
            x = x + 1

        # error msg
        else:
            print("Invalid log0 method; log0 method not applied. Use 'offset' or 'log1p'.")

        return x

    def _scale_gene_counts(self, gene_counts:pd.DataFrame):
        # handle log zero
        gene_counts = self._handle_log0(gene_counts)

        # take the (natural) log of all the values
        log_counts = np.log(gene_counts)

        # average each row (e.g., geometric average)
        geom_avg = log_counts.mean(axis=1)

        # filter out +-inf
        geom_avg_filt = geom_avg[(abs(geom_avg) != np.inf)]

        # subtract the average log value from the log(counts)
        log_ratio = log_counts.sub(geom_avg_filt, axis=0)

        # caclulate the median of the ratios for each sample
        log_ratio_median = log_ratio.median()

        # calculate scaling factors (e^log_ratio_median)
        scaling_factor = log_ratio_median.apply(lambda x: np.exp(x))

        # divide the original read counts by the scaling factors; return output
        return gene_counts.div(scaling_factor, axis=1)
    
    def _fc_gene_counts(self, gene_counts:pd.DataFrame, metadata:pd.DataFrame, metadata_ctrl:str):
        # get control barcodes
        ctrl_barcodes = metadata[metadata['type'] == metadata_ctrl]['barcode']

        # get mean
        ctrl_avg = gene_counts[ctrl_barcodes].mean(axis=1).values.reshape(-1, 1) # reshape to allow division

        # get fc
        return gene_counts / ctrl_avg

    def _graph_preprocessing(self, gene_counts:pd.DataFrame, relation:pd.DataFrame):
        # filter gene_counts by relation ensembl
        unique_ensg = pd.concat([relation[cols] for cols in ['ensembl1', 'ensembl2']]).unique().tolist()
        gene_counts = gene_counts.loc[unique_ensg,:]

        # get id maps
        node_id_map = {node: int(i) for i, node in enumerate(gene_counts.index)}

        # map relation ensembl id to nodes
        relation['idx1'] = relation['ensembl1'].map(node_id_map)
        relation['idx2'] = relation['ensembl2'].map(node_id_map)

        # replace ensembl with idx
        cols = relation.columns.to_list()
        cols = [col for col in cols if col not in ['pathway_name', 'ensembl1', 'ensembl2','idx1','idx2']]
        cols = ['pathway_name', 'idx1', 'idx2'] + cols
        relation = relation.loc[:,cols]

        return gene_counts, relation, node_id_map

    def _get_masks(self, relation:pd.DataFrame):
        # initialize empty dict
        mask_nodes = {}

        # iterate through df grouped by mask_id
        for mask_id, group in relation.groupby('pathway_name'):
            nodes = pd.concat([group['idx1'], group['idx2']]).unique() # get unique nodes in idx1 & idx2
            mask_nodes[mask_id] = nodes.tolist() # append to dict

        # list masks
        masks = [j for _,j in mask_nodes.items()]

        return masks
    
    def _filter_counts(self, gene_counts:pd.DataFrame, metadata:pd.DataFrame, y_col:str, drop:dict, max_subset:int):
            # drop cols by class
            if (drop != None) & (type(drop) == dict):
                for key, value in drop.items():
                    metadata = metadata[~metadata[key].isin(value)]

            # downsample
            if (max_subset != None) & (type(max_subset) == int):

                # helper fxn
                def downsample(group):
                    if len(group) > max_subset:
                        return group.sample(n=max_subset)
                    return group
                
                # apply downsampling
                metadata_grouped = metadata.groupby(y_col)
                metadata_grouped = metadata_grouped.apply(downsample, include_groups=False).reset_index(drop=True)

                # reappend y_col, reorder
                metadata = pd.merge(metadata_grouped, metadata[['barcode', y_col]], on='barcode', how='left')
                metadata = metadata[['barcode','type','subtype']]

            # apply filter
            gene_counts = gene_counts[metadata['barcode']]

            return gene_counts, metadata
    
    def _get_Xy(self, gene_counts:pd.DataFrame, metadata:pd.DataFrame, y_col:str):
        # get counts
        X = gene_counts.T
        y_labels = metadata[y_col]

        # format 
        X = X.values.astype(np.float32)
        y = pd.get_dummies(y_labels).values.astype(np.float32)
        y_labels = y_labels.unique().tolist()

        # add 3rd dim, e.g. reshape to [num_samples, num_nodes, 1]
        X = np.expand_dims(X, axis=-1)

        # to tensor
        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, y, y_labels

    def _get_dims(self, X:torch.Tensor, y:torch.Tensor, masks:list):
        num_samples, num_nodes, num_features = X.shape
        _, num_labels = y.shape
        num_masks = len(masks)

        return num_samples, num_nodes, num_features, num_labels, num_masks
    
    def __str__(self):

        out = ''
        width = 16 # print col width

        for name, variable in self.__dict__.items():
            # get variable shape
            if type(variable) == pd.core.frame.DataFrame:
                shape = variable.shape
            elif type(variable) == torch.Tensor or type(variable) == np.ndarray:
                shape = tuple([i for i in variable.shape])
            elif (type(variable) == list) or (type(variable) == dict):
                shape = len(variable)
            elif (type(variable) == int) or (type(variable) == str):
                shape = variable
            else:
                shape = None

            # append shape if applicable
            if shape != None:
                try:
                    out += f'{name:<{width}} {str(shape):<{width}} {type(variable).__name__} ({variable.device.__str__()})\n'
                except:
                    out += f'{name:<{width}} {str(shape):<{width}} {type(variable).__name__}\n'
            else:
                out += f'{name:<{width}} {type(variable).__name__}\n'

        # return string
        return out

