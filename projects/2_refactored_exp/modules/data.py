import inspect
import numpy as np
import pandas as pd
import torch

from modules.utils import capture_kwargs, vprint, dict_summary
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

from pandas import DataFrame
from typing import Literal, Optional, Union

class KEGG():
    def __init__(self, relation_filepath:str, counts_data=None, get_counts:bool=True, coo_pathway:bool=False, verbose:bool=True):
        # get kwargs
        sig = inspect.signature(type(self).__init__)
        self._orig_kwargs = capture_kwargs(sig, self, relation_filepath, counts_data, get_counts, coo_pathway, verbose)

        vprint('# #### KEGG() ####', verbose=verbose)
        ensg_filter = counts_data.ensg_complete if counts_data is not None else None
        self.relation, self.ensg, self.pathway_labels = self._read_relation(relation_filepath, ensg_filter)
        self.edge_index, self.edge_attr, self.edge_labels = self._get_edges(self.relation)
        self.pathway_index = self._get_hyperedges(self.relation, coo_pathway)
        # del self.relation
        vprint(self, verbose=verbose)

        if (counts_data is not None) & get_counts:
            counts_data.get_counts(ensg_filter=self.ensg)

    def __str__(self):
        return dict_summary(self.__dict__)
    
    def _read_relation(self, relation_filepath:str, ensg_filter:Optional[list[int]]):
        # get relation df
        relation = pd.read_csv(relation_filepath)

        # get list of unique pw
        pathway = relation['pathway_name'].unique().tolist()

        # get list of unique ensg
        ensg = pd.concat([relation['ensembl1'], relation['ensembl2']]).unique().tolist()
        if ensg_filter is not None: # filter if provided
            ensg = list(set(ensg_filter) & set(ensg))
        ensg = sorted(ensg) # sort

        # map to idx
        node_id_map = {node: int(i) for i, node in enumerate(ensg)}
        set_id_map = {_set: int(i) for i, _set in enumerate(pathway)}
        relation['node_i_idx'] = relation['ensembl1'].map(node_id_map)
        relation['node_j_idx'] = relation['ensembl2'].map(node_id_map)
        relation['set_idx'] = relation['pathway_name'].map(set_id_map)

        # replace ensembl with idx
        cols = relation.columns.to_list()
        cols = [col for col in cols if col not in ['pathway_name', 'ensembl1', 'ensembl2', 'set_idx', 'node_i_idx','node_j_idx']]
        cols = ['set_idx', 'node_i_idx', 'node_j_idx'] + cols
        relation = relation.loc[:,cols]

        return relation, ensg, pathway
    
    def _get_edges(self, relation:DataFrame):
        # get unique
        relation = relation.drop(columns='set_idx').groupby(['node_i_idx','node_j_idx'], as_index=False).any()

        # get indices as tensor
        edge_index = relation[['node_i_idx', 'node_j_idx']]
        edge_index = torch.tensor(edge_index.values.T, dtype=torch.long)

        # get attr as tensor
        edge_attr = relation.drop(columns=['node_i_idx', 'node_j_idx'])
        edge_labels = edge_attr.columns.tolist()
        edge_attr = torch.tensor(edge_attr.astype(int).values)
        
        return edge_index, edge_attr, edge_labels
    
    def _get_hyperedges(self, relation:DataFrame, coo_pathway:bool):
        # get unique node-set indices
        hyperedge_index = pd.concat([
            relation[['set_idx', 'node_i_idx']].rename(columns={'node_i_idx': 'node_idx'}).dropna().astype(int),
            relation[['set_idx', 'node_j_idx']].rename(columns={'node_j_idx': 'node_idx'}).dropna().astype(int)
        ]).drop_duplicates()

        # convert to tensor
        hyperedge_index = torch.tensor(hyperedge_index.values.T)

        # dense if not coo
        if not coo_pathway:
            # hyperedge_index: torch.Size([2, num_edges]) with rows [set_idx, node_idx]
            set_indices = hyperedge_index[0]  # shape: (num_edges,)
            node_indices = hyperedge_index[1]  # shape: (num_edges,)

            num_sets = set_indices.max().item() + 1
            num_nodes = node_indices.max().item() + 1

            # Create a dense incidence matrix
            hyperedge_index = torch.zeros((num_nodes, num_sets), dtype=torch.float32)
            hyperedge_index[node_indices, set_indices] = 1.0

        return hyperedge_index

class TCGA():
    def __init__(
        self, 
        # paths
        tcga_project:str, 
        tcga_dir:str,

        # count labels
        gene_name_path:Optional[str]=None,
        keep_noname:bool=False,

        # metadata
        type_col:Optional[str]=None,
        subtype_col:Optional[str]=None,
        drop:Optional[list]=None,

        # auto get_counts
        get_counts:Union[bool, list[int]]=False, # accepts ensg_filter
        verbose:bool=True
    ):
        # get kwargs
        sig = inspect.signature(type(self).__init__)
        self._orig_kwargs = capture_kwargs(sig, self, tcga_project, tcga_dir, gene_name_path, keep_noname, type_col, subtype_col, drop, get_counts, verbose)

        # paths for read
        self.counts_path = Path(tcga_dir) / f'TCGA-{tcga_project}_gene_counts.csv'
        self.metadata_path = Path(tcga_dir) / f'TCGA-{tcga_project}_metadata.csv'
        self.gene_name_path = gene_name_path
        
        # get metadata, process
        self.metadata_complete = pd.read_csv(self.metadata_path).drop(columns='Unnamed: 0')
        self.metadata, self.y, self.y_labels = self._get_metadata(self.metadata_complete, type_col, subtype_col, drop)

        # get data
        self.ensgv, self.ensg_complete = self._get_ensg(keep_noname)

        # get counts
        if get_counts is True:
            self.get_counts(ensg_filter=self.ensg_complete, verbose=verbose)
        elif get_counts is not False:
            self.get_counts(ensg_filter=get_counts, verbose=verbose)
    
    def _get_metadata(self, df_complete:DataFrame, type_col:Optional[str], subtype_col:Optional[str], drop:Union[str,list[str],None]) -> DataFrame:
        '''
        Gets specified type and subtype from metadata_complete.
        Type defaults to 'name' if tissue_type='Tumor', else 'tissue_type' (e.g. 'Normal')
        Subtype default disabled. If provided, nan subtype defualts to 'sample_type'

        Example:
        name: ['Breast Invasive Carcinoma']
        tissue_type: ['Tumor', 'Normal']
        sample_type: ['Primary Tumor', 'Solid Tissue Normal', 'Metastatic']
        '''
        # get cols
        if isinstance(type_col, str):
            type_col = df_complete[type_col]
        else: # None (default): use 'name' if Tumor, else use Normal
            type_col = df_complete.apply(lambda row: row['name'] if row['tissue_type'] == 'Tumor' else row['tissue_type'], axis=1)

        # convert to subtype if applicable
        if isinstance(subtype_col, str):
            type_col = df_complete[subtype_col].fillna(df_complete['sample_type'])

        # compile df
        df = pd.DataFrame({'barcode': df_complete['barcode'], 'type': type_col})

        # drop y_col types if applicable
        if isinstance(drop, str):
            drop = [drop] # conv to list
        if isinstance(drop, list):
            df = df[~df['type'].isin(drop)]

        # get y, y_labels
        cat = pd.Categorical(df['type'])
        y = torch.tensor(cat.codes)
        y_labels = cat.categories.tolist()
        
        return df, y, y_labels

    def _get_ensg(self, keep_noname:bool) -> DataFrame:
        # read gene ID col, squeeze to series -> list
        df = pd.read_csv(self.counts_path, usecols=[0], names=['ensgv'], header=0)#.squeeze().tolist()
        df['ensg'] = df['ensgv'].str.split('.').str[0]

        # add gene names if applicable
        if self.gene_name_path is None:
            ensg = df['ensg'].drop_duplicates().tolist()
        else:
            names = pd.read_csv(self.gene_name_path).drop_duplicates('ensg')
            df = pd.merge(left=df, right=names, how='left', on='ensg')
            ensg = df['ensg'].drop_duplicates().tolist() if keep_noname else df['ensg'][~df['name'].isna()].drop_duplicates().tolist()
            
        return df, ensg

    def _read_counts(self, ensg_filter:Optional[list[int]]) -> DataFrame:
        # filter genes if applicable
        if ensg_filter is not None:
            keeprows = self.ensgv['ensg'].isin(set(ensg_filter))
            skiprows = [i+1 for i, keep in enumerate(keeprows) if not keep]
        else:
            skiprows = None

        # read, format idx, transpose
        df = pd.read_csv(self.counts_path, skiprows=skiprows, index_col=0) # read
        df.index = df.index.str.split('.').str[0] # strip ensg vers
        df.index.name = 'ensg' # name index
        df = df[~df.index.duplicated(keep='first')] # drop duplicates
        df = df.T # transpose

        return df

    def get_counts(self, ensg_filter:Optional[list[int]]=None, verbose:bool=True):
        # get counts
        self.counts = self._read_counts(ensg_filter)

        # match metadata filters if applicable
        self.counts = self.counts.loc[self.counts.index.isin(self.metadata['barcode'])]

        # values as tensor
        self.x = torch.tensor(self.counts.values).unsqueeze(-1)

        # get x labels (names)
        ensg = self.counts.columns.tolist()

        if 'name' in self.ensgv.columns:
            ensg2name = dict(zip(self.ensgv['ensg'], self.ensgv['name']))
            self.x_labels = [ensg2name.get(i) for i in ensg]
        else:
            self.x_labels = ensg

        vprint('# #### TCGA() ####', verbose=verbose)
        vprint(dict_summary(self.__dict__), verbose=verbose)

class DataWrapper():
    def __init__(self, counts_data, edge_data=None, pathway_data=None, verbose:bool=True):
        # get params
        self.counts_params = counts_data._orig_kwargs
        self.edge_params = edge_data._orig_kwargs
        self.pathway_params = None if (edge_data == pathway_data) else pathway_data._orig_kwargs

        vprint('# #### DataWrapper() ####', verbose=verbose)
        # counts
        self.x = counts_data.x
        self.y = counts_data.y.to(torch.long)
        self.x_labels = counts_data.x_labels
        self.y_labels = counts_data.y_labels

        # edges
        if edge_data is not None:
            self.edge_index = edge_data.edge_index
            self.edge_attr = edge_data.edge_attr
            self.edge_labels = edge_data.edge_labels

        # pathways
        if pathway_data is not None:
            self.pathway_index = pathway_data.pathway_index
            self.pathway_labels = pathway_data.pathway_labels

        # dims
        self.num_samples, self.num_nodes, self.num_node_features = self.x.shape
        self.num_classes = self.y.unique().size(0)
        self.num_edges, self.num_edge_features = self.edge_attr.shape if edge_data is not None else (None, None)
        self.num_pathways = len(self.pathway_labels)

        vprint(self, verbose=verbose)

    def __str__(self):
        return dict_summary(self.__dict__)

class GraphDataset(InMemoryDataset):
    def __init__(self, counts_data, edge_data, pathway_data=None, verbose:bool=True):
        super().__init__()

        # construct wrapper
        self.wrapper = DataWrapper(counts_data, edge_data, pathway_data, verbose)

        # extract data for graph dataset
        self.x = self.wrapper.x.to(torch.float)
        self.y = self.wrapper.y
        self.edge_index = self.wrapper.edge_index
        self.edge_attr = self.wrapper.edge_attr

        # process data
        self.data, self.slices = self._process_data()

        # print summary
        if verbose == True:
            self.print_dims()

    def _process_data(self): 
        data_list = []
        num_samples = self.x.size(0)

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

            # append to list
            data_list.append(data_entry)

        # collate list
        data, slices = self.collate(data_list)

        return data, slices
    
    def print_dims(self):
        # get first graph in dataset
        data = self[0]

        # summmarize graph information
        data_summary = {
            'Average node degree': data.num_nodes / data.num_edges,
            'Has isolated nodes': data.has_isolated_nodes(),
            'Has self-loops': data.has_self_loops(),
            'Directionality': 'directed' if data.is_directed() else 'undirected'
        }

        # create msg
        out = '\n# #### GraphDataset(), Summary ####\n'
        out += dict_summary(data_summary)

        # print msg
        print(out)

def get_toy_databatch(dataset:DataWrapper, device:str, batch_size:int = 64):
    # init seed, generator
    generator = torch.Generator(device=device)

    # generate loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    # get batches
    batches = [(step, data) for step, data in enumerate(loader)]

    # return batch 0 data
    return batches[0][1] 