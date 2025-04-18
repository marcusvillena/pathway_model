import pandas as pd
import numpy as np
import torch

from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from .utils import vprint, dict_summary

# typing
from pandas import DataFrame
from typing import Literal


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

class Preprocessor():
    def __init__(
        self, 
        # paths
        tcga_project:str, 
        tcga_dir:str, 
        relation_filepath:str, 

        # sample preprocessing
        metadata_subtype_col:str='', 
        log0_method:Literal['log1p','offset']='log1p',
        scale_gene_counts:bool=True, 

        # filter, resample, etc.
        y_col:str='type',
        y_format:Literal['index','onehot']='index',
        drop:list[str]=None,
        max_subset:int=None,
        edge_method:Literal['in','out']='in',
        verbose:bool=True
    ):
        '''
        Class to format and preprocess TCGA data, TCGA metadata, and KEGG relation data.
        '''
        
        vprint('# #### Preprocessor() ####', verbose=verbose)
        
        # assign params to inst vars
        self.log0_method = log0_method

        # read files into df
        gene_counts = self._read_gene_counts(tcga_project, tcga_dir)
        metadata = self._read_metadata(tcga_project, tcga_dir, metadata_subtype_col)
        relation = pd.read_csv(relation_filepath)

        # data preprocessing
        if scale_gene_counts == True:
            gene_counts = self._scale_gene_counts(gene_counts)

        # preprocess gene_counts and relation into graph data
        gene_counts, relation, node_id_map = self._graph_preprocessing(gene_counts, relation)

        # get masks, flatten relation
        masks = self._get_masks(relation)
        relation = relation.drop(columns='pathway_name').groupby(['idx1','idx2'], as_index=False).any()

        # filter counts by class (drop classes, downsample) if applicable
        gene_counts, metadata = self._filter_counts(gene_counts, metadata, y_col, drop, max_subset)

        # get xy
        x, y, y_labels = self._get_xy(gene_counts, metadata, y_col, y_format)

        # get class_weights
        self.class_weights = y.shape[0]/y.sum(dim=0)

        # get edge information
        self.edge_index, self.edge_attr = self._get_edges(relation, edge_method)

        # assign instance variables
        self.gene_counts = gene_counts
        self.metadata = metadata
        self.relation = relation
        self.node_id_map = node_id_map
        self.masks = masks
        self.x = x
        self.y = y
        self.y_labels = y_labels

        # get dims
        self._get_dims(y_format)
        vprint(self, verbose=verbose)

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

    def _get_xy(self, gene_counts:pd.DataFrame, metadata:pd.DataFrame, y_col:str, y_format:Literal['index','onehot']):
        # format x
        x = gene_counts.T
        x = x.values.astype(np.float32)
        x = np.expand_dims(x, axis=-1) # reshape to (num_samples, num_nodes, 1)
        x = torch.tensor(x)

        # format y
        y_labels = metadata[y_col]
        y = pd.get_dummies(y_labels).values.astype(np.float32)
        y_labels = y_labels.unique().tolist()        
        y = torch.tensor(y)

        # convert y to index, if applicable
        y = y.argmax(dim=1) if y_format == 'index' else y
        
        return x, y, y_labels

    def _get_edges(self, relation:DataFrame, edge_method:Literal['in','out'], source:str='idx1', target:str='idx2'):
        # get edge information >> (num_edges, 2)
        if edge_method == 'in':
            edge_index = relation[[target, source]]
        elif edge_method == 'out':
            edge_index = relation[[source, target]]
        
        # transpose, convert to tensor >> (2, num_edges) 
        edge_index = torch.tensor(edge_index.values.T, dtype=torch.int64)

        # get edge attr
        attr_cols = relation.columns.drop([source, target])
        if len(attr_cols) > 0:
            edge_attr = torch.tensor(relation[attr_cols].values, dtype=torch.int64)
        else:
            edge_attr = None

        return edge_index, edge_attr

    def _get_dims(self, y_format:Literal['index','onehot']):
        # x
        self.num_samples, self.num_nodes, self.num_node_features = self.x.shape

        # y
        if y_format == 'onehot':
            _, self.num_classes = self.y.shape
        else:
            self.num_classes = self.y.unique().size(0)

        # masks
        self.num_masks = len(self.masks)

        # edges
        self.num_edges, self.num_edge_features = self.edge_attr.shape
    
    def __str__(self):
        return dict_summary(self.__dict__)

class GraphDataset(InMemoryDataset):
    def __init__(self, preprocessor:Preprocessor, verbose:bool=True, *args, **kwargs):
        # assign instance variables
        self.x = preprocessor.x
        self.y = preprocessor.y
        self.edge_index = preprocessor.edge_index
        self.edge_attr = preprocessor.edge_attr

        # super args, kwargs
        super().__init__(*args, **kwargs)

        # process data
        self.data, self.slices = self.process_data()

        # print summary
        if verbose == True:
            self.print_dims()

    def process_data(self):
        data_list = []
        num_samples = self.x.size(0)

        for i in range(num_samples):
            data = Data(
                x=self.x[i],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=self.y[i]
            )

            data_list.append(data)

        return self.collate(data_list)

    def print_dict(self):
        # get first graph in dataset
        data = self[0] 

        # format msg
        out = '# #### GraphDataset(), Dataset (Dict) ####\n'
        out += dict_summary(self.__dict__)
        out += '\n# #### GraphDataset(), Data (Dict) ####\n'
        out += dict_summary(data.__dict__)

        # print msg
        print(out)

    def print_dims(self):
        # get first graph in dataset
        data = self[0]

        # get dims
        dataset_dims = {
            'num_graphs (len)':len(self),
            'num_node_features':self.num_node_features,
            'num_edge_features':self.num_edge_features,
        }
        data_dims = {
            'num_nodes':data.num_nodes,
            'num_edges':data.num_edges,
            'num_node_features':data.num_node_features,
            'num_edge_features':data.num_edge_features,
        }
        data_summary = {
            'Average node degree': data.num_nodes / data.num_edges,
            'Has isolated nodes': data.has_isolated_nodes(),
            'Has self-loops': data.has_self_loops(),
            'Directionality': 'directed' if data.is_directed() else 'undirected'
        }

        # create msg
        out = '# #### GraphDataset(), Dataset ####\n'
        out += dict_summary(dataset_dims)
        out += '\n# #### GraphDataset(), Data ####\n'
        out += dict_summary(data_dims)
        out += '\n# #### GraphDataset(), Summary ####\n'
        out += dict_summary(data_summary)

        # print msg
        print(out)

        


