#### mv packages ####
from .utils import vprint, dict_summary

#### packages ####
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE

#### typing ####
from torch import Tensor
from pandas import DataFrame
from typing import Literal

class Preprocessor():
    def __init__(
        self, 
        # paths
        tcga_project:str, 
        tcga_dir:str, 
        relation_filepath:str, 
        metadata_subtype_col:str='', 

        # fc
        apply_fold_change:bool=False,
        metadata_ctrl:str='Normal',

        # sample preprocessing
        log0_method:Literal['log1p','offset']='log1p',
        apply_DESeq_norm:bool=True, 
        log_transform:bool=True,
        scale_method:Literal['robust', 'standard', None]=None,

        # filter, resample, etc.
        y_col:str='type',
        y_format:Literal['index','onehot']='index',
        drop:list[str]=None,
        max_subset:int=None,
        class_wt_method:Literal['inverse','inverse_log']='inverse_log',
        edge_method:Literal['in','out']='in',
        verbose:bool=True,
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
        if apply_DESeq_norm == True:
            gene_counts = self._DESeq_norm(gene_counts)
            # self.deseq = gene_counts

        if apply_fold_change == True:
            gene_counts = self._fold_change(gene_counts, metadata, metadata_ctrl)

        if log_transform == True:
            gene_counts = np.log(self._handle_log0(gene_counts))

        if scale_method != None:
            gene_counts = self._scale_gene_counts(gene_counts, scale_method)

        # preprocess gene_counts and relation into graph data
        gene_counts, relation, node_id_map = self._graph_preprocessing(gene_counts, relation)

        # get masks, flatten relation
        mask, mask_list = self._get_masks(relation)
        relation = relation.drop(columns='pathway_name').groupby(['idx1','idx2'], as_index=False).any()

        # filter counts by class (drop classes, downsample) if applicable
        gene_counts, metadata = self._filter_counts(gene_counts, metadata, y_col, drop, max_subset)

        # get xy
        x, y, y_labels = self._get_xy(gene_counts, metadata, y_col, y_format)

        # get class_weights
        self.class_weights = self._get_class_weights(y, class_wt_method)

        # get edge information
        self.edge_index, self.edge_attr = self._get_edges(relation, edge_method)

        # assign instance variables
        self.gene_counts = gene_counts
        self.metadata = metadata
        self.relation = relation
        self.node_id_map = node_id_map
        self.mask_list = mask_list
        self.mask = mask
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
    
    def _fold_change(self, gene_counts:pd.DataFrame, metadata:pd.DataFrame, metadata_ctrl:str):
        # get control barcodes
        ctrl_barcodes = metadata[metadata['type'] == metadata_ctrl]['barcode']

        # get mean
        ctrl_avg = gene_counts[ctrl_barcodes].mean(axis=1).values.reshape(-1, 1) # reshape to allow division

        # get fc
        return gene_counts / ctrl_avg
    
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

    def _DESeq_norm(self, gene_counts:pd.DataFrame):
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
    
    def _scale_gene_counts(self, gene_counts:pd.DataFrame, scale_method:Literal['robust', 'standard', None]=None):
        # return if none
        if scale_method == None:
            return gene_counts

        # select scaler        
        if scale_method == 'robust':
            scaler = RobustScaler()
        elif scale_method == 'standard':
            scaler = StandardScaler()

        # scale data
        scaled_values = scaler.fit_transform(gene_counts.T).T

        # convert to df
        gene_counts = pd.DataFrame(
            scaled_values,
            index=gene_counts.index,
            columns=gene_counts.columns
        )

        return gene_counts
    
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
        mask_list = [j for _,j in mask_nodes.items()]

        # get dims
        num_masks = len(mask_list)
        num_nodes = len(set([node for mask in mask_list for node in mask]))
        
        # convert to tensor
        mask_tensor = torch.zeros([num_nodes, num_masks])
        for mask in range(num_masks):
            for node in mask_list[mask]:
                mask_tensor[node][mask] = 1

        return mask_tensor, mask_list
    
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

    def _get_class_weights(self, y, class_wt_method:Literal['inverse','inverse_log']):
        num_classes = y.max().item() + 1
        
        # count per class
        count = torch.bincount(y, minlength=num_classes)

        # get class weights
        if class_wt_method=='inverse_log':
            class_weights = 1/torch.log1p(count)
        else:
            class_weights = 1/count

        # normalize
        class_weights = num_classes * class_weights / class_weights.sum()

        return class_weights

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
            edge_attr = torch.tensor(relation[attr_cols].values, dtype=torch.float32)
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
        self.num_masks = len(self.mask_list)

        # edges
        self.num_edges, self.num_edge_features = self.edge_attr.shape
    
    def __str__(self):
        return dict_summary(self.__dict__)
    
class GraphDataset(InMemoryDataset):
    def __init__(self, preprocessor:Preprocessor, laplacian_eigs:int=16, verbose:bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = preprocessor.x
        self.y = preprocessor.y
        self.edge_index = preprocessor.edge_index
        self.edge_attr = preprocessor.edge_attr
        self.laplacian_eigs = laplacian_eigs
        self.device = self.x.device    

        # process data
        self.data, self.slices = self._process_data()

        # print summary
        if verbose == True:
            self.print_dims()

    def _add_laplacian_PE(self):
        torch.set_default_device('cpu')

        # Step 1: Move edge_index safely to CPU
        temp_x = self.x.cpu()
        temp_edge_index = self.edge_index.cpu()

        # Step 2: Build a safe temporary graph
        num_nodes = temp_edge_index.max().item() + 1
        temp_graph = Data(
            x=temp_x[0],
            edge_index=temp_edge_index,
            num_nodes=num_nodes
        )

        # Step 3: Apply Laplacian PE
        laplacian = AddLaplacianEigenvectorPE(k=self.laplacian_eigs)
        temp_graph = laplacian(temp_graph)

        # Step 4: Move laplacian_pe to correct device
        temp_graph.laplacian_eigenvector_pe = temp_graph.laplacian_eigenvector_pe.to(self.device)

        # Step 5: Return laplacian_pe
        laplacian_pe = temp_graph.laplacian_eigenvector_pe
        
        torch.set_default_device(self.device)

        return laplacian_pe

    def _process_data(self): 
        data_list = []
        num_samples = self.x.size(0)

        # generate laplacian PE
        laplacian_pe = self._add_laplacian_PE()

        # for each sample
        for i in range(num_samples):
            
            # create a graph (Data) per sample
            data_entry = Data(
                x=self.x[i],
                y=self.y[i],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
            )

            # add sample id
            data_entry.sample_id = i

            # add laplacian pe
            data_entry.laplacian_pe = laplacian_pe

            # append to list
            data_list.append(data_entry)

        # collate list
        data, slices = self.collate(data_list)

        return data, slices

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

def get_toy_databatch(dataset:GraphDataset, generator, batch_size:int = 64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    batches = [(step, data) for step, data in enumerate(loader)]
    return batches[0][1] # return batch 0 data