import torch
import subprocess
import numpy as np
import pandas as pd

from torch import Tensor
from torch_geometric.data import Batch
from typing import Literal, Optional, Union

def vprint(*objects, verbose=True, **kwargs):
    if verbose==True:
        print(*objects, **kwargs)

def dict_summary(_dict:dict, width:int=24):
    # init str
    out = ''

    for key, value in _dict.items():
        # get variable shape
        if type(value) == pd.DataFrame:
            shape = value.shape
        elif type(value) in [torch.Tensor, np.ndarray]:
            shape = tuple([i for i in value.shape])
        elif type(value) in [list, dict]:
            shape = len(value)
        elif type(value) in [int, str, bool]:
            shape = value
        elif type(value) == float:
            shape = f'{value:.4f}'
        else:
            shape = None

        # append shape if applicable
        if shape != None:
            try:
                out += f'# {key:<{width}} {str(shape):<{width}} {type(value).__name__} ({value.device.__str__()})\n'
            except:
                out +=  f'# {key:<{width}} {str(shape):<{width}} {type(value).__name__}\n'
        else:
            out += f'# {key:<{width}} {type(value).__name__}\n'

    return out

def reshape_x(x:Union[Tensor, Batch], to:Literal['b,n,f','b*n,f','b,n*f'], batch_size:Optional[int]=None, num_nodes:Optional[int]=None, num_node_features:Optional[int]=None, return_dims:bool=False):
    '''
    detects x of size (b,n,f), (b*n,f), or (b,n*f) and returns desired view
    '''
    # if batch
    if hasattr(x, 'x'):
        batch_size = x.batch_size
        num_node_features = x.num_node_features
        x = x.x
        
    # ensure supported dim
    assert x.dim() in (2,3), f'unsupported x.dim(): {x.dim()}'

    # b,n,f all known
    if (batch_size is not None) and (num_nodes is not None) and (num_node_features is not None):
        pass # do nothing
    elif x.dim() == 3:
        batch_size, num_nodes, num_node_features = x.shape

    # one unknown (dim = 2)
    else:
        # find num_nodes
        if (batch_size is not None) and (num_node_features is not None):
            if x.shape[-1] == num_node_features: # b*n,f case
                num_nodes = int(x.shape[0]//batch_size)
            else: # b,n*f case
                num_nodes = int(x.shape[-1]//num_node_features)

        # find batch_size
        elif (num_nodes is not None) and (num_node_features is not None):
            if x.shape[-1] == num_node_features: # b*n,f case
                batch_size = int(x.shape[0]//num_nodes)
            else: # b,n*f case
                batch_size = x.shape[0]

        # find num_node_features
        elif (batch_size is not None) and (num_nodes is not None):
            if x.shape[0] == batch_size: # b,n*f case
                num_node_features = int(x.shape[-1]//num_nodes)
            else: # b*n,f case
                num_nodes = x.shape[-1]

        # not enough information
        assert sum(p is not None for p in [batch_size, num_nodes, num_node_features]) >= 2, 'two of [batch_size, num_nodes, num_node_features] must be provided'

    # reshape
    if to == 'b,n,f':
        x = x.reshape(batch_size, num_nodes, num_node_features)
    elif to == 'b*n,f':
        x = x.reshape(batch_size * num_nodes, num_node_features)
    else: # 'b,n*f
        x = x.reshape(batch_size, num_nodes * num_node_features)

    return (x, batch_size, num_nodes, num_node_features) if return_dims else x

class Devices():
    def __init__(self, verbose:bool=True):
        self.verbose = verbose

        # devices
        self.info = self._get_available_devices()
        self.list = [key for key in self.info.keys()]

        # cuda devices
        if torch.cuda.is_available():
            self.gpu_info = self._cuda_list_gpus() # (device, name, free mem.)
            self.gpu_list = [i[0] for i in self.gpu_info] # gpus sorted by most free mem.
        elif torch.backends.mps.is_available():
            self.gpu_info = [('mps')]
            self.gpu_list = ['mps']
        else:
            self.gpu_info = self.gpu_list = []

    def _get_available_devices(self):
        # init list with cpu
        available = {'cpu':''}

        # append cuda if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                available[f'cuda:{i}'] = f'{torch.cuda.get_device_name(i)}'

        # append mps if available
        if torch.backends.mps.is_available():
            available['mps'] = None

        return available
    
    def _cuda_check_memory(self):
        # define cli command
        command = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"

        # get cli result
        result = subprocess.check_output(command, shell=True, text=True)

        # format result into list of ints
        free_memory = [int(i) for i in result.strip().split('\n')]

        return free_memory
    
    def _cuda_list_gpus(self):
        # init gpu list
        gpu_list = []

        # get gpu free memory
        free_memory = self._cuda_check_memory()

        # get info
        for i in range(torch.cuda.device_count()):
            # torch device and name
            device = f'cuda:{i}'
            name = f'{torch.cuda.get_device_name(i)}'

            # format to tuple append to gpu list
            gpu = (device, name, free_memory[i])
            gpu_list.append(gpu)

        # sort gpu_list by free memory
        gpu_list = sorted(
            gpu_list,
            key = lambda gpu: gpu[2],
            reverse=True
        )

        return gpu_list 
    
    def set_device(self, device):
        vprint('# #### Device() ####', verbose=self.verbose)

        # set device, generator
        torch.set_default_device(device)
        generator = torch.Generator(device=device)

        # print
        vprint(f'# device = {torch.get_default_device().__str__()}\n', verbose=self.verbose)
        return device, generator
    
    def auto_set_device(self, drop:list=[]):
        # check device to use, cuda > mps > cpu
        if torch.cuda.is_available():
            drop_gpu_list = [gpu for gpu in self.gpu_list if gpu not in drop]
            device = drop_gpu_list[0] # use gpu with highest free memory
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        # set device
        device, generator = self.set_device(device)

        return device, generator