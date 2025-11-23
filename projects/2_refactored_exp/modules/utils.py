import copy
import functools
import inspect
import itertools
import torch
import torch.nn as nn
import subprocess
import numpy as np
import pandas as pd

# typing
from torch import Tensor
from torch_geometric.data import Batch, Data
from typing import Callable, Literal, Optional, Type, Union

def tsoftmax(input:Tensor, temperature:float=1, dim:int=None, dtype:Optional[torch.dtype]=None):
    if temperature is None:
        temperature = 1
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    
    return torch.softmax(input / temperature, dim=dim)

# general
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

# training
def clean_name(name): # for t.grid()
    # convert to str
    if isinstance(name,float):
        name = f'{name:.5e}' # sci not.
        name = name.replace('e-0','e-')
        name = name.replace('e+0','e')
        mantissa, exp = name.split('e')
        mantissa = mantissa.rstrip('0').rstrip('.')
        name = f'{mantissa}e{exp}'
    if isinstance(name, str):
        name.capitalize()
    else:
        name = f'{name}'

    # clean str
    name = name.replace('.','p')
    name = name.replace('_','')
    
    return name

# models
def build_hidden_dims(embed_dim:int, hidden_dims:Union[int, list[int], None]) -> Optional[list[int]]:
    # none case
    if (hidden_dims is None) or isinstance(hidden_dims, list):
        return hidden_dims

    # int case: use as depth
    elif isinstance(hidden_dims, int):
        return [embed_dim] * hidden_dims

    # error case
    raise TypeError(f"{hidden_dims} must be a int or list[int], got {type(hidden_dims)}")  

def clone_or_init(
   name:str,
   obj:Union[nn.Module, Type[nn.Module]],
   base_class:Type[nn.Module],
   builder: Callable[[Type[nn.Module]], nn.Module]
) -> nn.Module:
   # if (predefined) instance
   if isinstance(obj, base_class):
      if callable(getattr(obj, "clone", None)): 
         return obj.clone() # clone if applicable
      return copy.deepcopy(obj) # else deepcopy

   # elif (not defined) class
   elif isinstance(obj, type) and issubclass(obj, base_class):
      return builder(obj)

   # else error case
   raise TypeError(
      f"{name} must be a {base_class.__name__} instance or subclass, got {type(obj)}"
   )   

def input_to_dict(input):
    if isinstance(input, Tensor): # x (Tensor) only
        data = {'x':input}
    elif isinstance(input, Data): # PyG Data or DataBatch
        data = {key: getattr(input, key) for key in input.keys()}
    elif isinstance(input, dict): # predefined dict
        data = input
    else:
        raise TypeError(f'unsupported input type: {type(input)}')
    return data

def reshape(x:Union[Tensor, Batch], to:Literal['b,n,f','b*n,f','b,n*f'], batch_size:Optional[int]=None, num_nodes:Optional[int]=None, num_features:Optional[int]=None, return_dims:bool=False):
    '''
    detects x of size (b,n,f), (b*n,f), or (b,n*f) and returns desired view
    '''
    # if batch
    if hasattr(x, 'x'):
        batch_size = x.batch_size
        num_features = x.num_node_features
        x = x.x
        
    # ensure supported dim
    assert x.dim() in (2,3), f'unsupported x.dim(): {x.dim()}'

    # b,n,f all known
    if (batch_size is not None) and (num_nodes is not None) and (num_features is not None):
        pass # do nothing
    elif x.dim() == 3:
        batch_size, num_nodes, num_features = x.shape

    # one unknown (dim = 2)
    else:
        # find num_nodes
        if (batch_size is not None) and (num_features is not None):
            if x.shape[-1] == num_features: # b*n,f case
                num_nodes = int(x.shape[0]//batch_size)
            else: # b,n*f case
                num_nodes = int(x.shape[-1]//num_features)

        # find batch_size
        elif (num_nodes is not None) and (num_features is not None):
            if x.shape[-1] == num_features: # b*n,f case
                batch_size = int(x.shape[0]//num_nodes)
            else: # b,n*f case
                batch_size = x.shape[0]

        # find num_features
        elif (batch_size is not None) and (num_nodes is not None):
            if x.shape[0] == batch_size: # b,n*f case
                num_features = int(x.shape[-1]//num_nodes)
            else: # b*n,f case
                num_nodes = x.shape[-1]

        # not enough information
        assert sum(p is not None for p in [batch_size, num_nodes, num_features]) >= 2, 'two of [batch_size, num_nodes, num_features] must be provided'

    # reshape
    if to == 'b,n,f':
        x = x.reshape(batch_size, num_nodes, num_features)
    elif to == 'b*n,f':
        x = x.reshape(batch_size * num_nodes, num_features)
    else: # 'b,n*f
        x = x.reshape(batch_size, num_nodes * num_features)

    return (x, batch_size, num_nodes, num_features) if return_dims else x

def filter_kwargs(func):
    '''
    decorator/wrapper for safe_call. 

    for functions:
        filter_kwargs(func)(*args, **kwargs)

    for class.__init__ (constructors):
        filter_kwargs(class)(*args, **kwargs)
        this wraps __init__

    for class.__call__ (callable instances): 
        filter_kwargs(class(*args, **kwargs)) 
        this wraps __call__ e.g. inits the class (no filter), and wraps its call/forward fxn

    for both __init__ and __call__:
        filter_kwargs(filter_kwargs(class)(*args,**kwargs))
    '''
    
    # get list of args
    sig = inspect.signature(func)

    # check if accepts args, kwargs
    accepts_args = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL 
        for p in sig.parameters.values()
    )
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )

    # get number of positional args/kwargs to keep
    if not accepts_args:
        num_pos = sum(
            1 for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY, 
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        )
    else:
        num_pos = None

    # get valid keys (ignore positional only)
    if not accepts_kwargs:
        valid_keys = {
            name for name, p in sig.parameters.items()
            if p.kind is not inspect.Parameter.POSITIONAL_ONLY
        }
    else:
        valid_keys = None

    # build wrapper (called on each call)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # filter args (remove extra pos arg/kwargs), and extra args provided
        if not accepts_args and len(args) > num_pos:
            args = args[:num_pos]

        # filter kwargs (remove positional-only), and kwargs not empty
        if not accepts_kwargs and kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        return func(*args, **kwargs)

    return wrapper

def capture_kwargs(sig:inspect.Signature, *args, **kwargs):
    # bind args/kwargs passed to func (sig)
    bound = sig.bind(*args, **kwargs)

    # fill in defaults for remaining
    bound.apply_defaults()

    # get original kwargs
    orig_kwargs = {}
    for name, value in bound.arguments.items(): # dict

        # skip 'self' instance
        if name == 'self': continue

        # skip *args and **kwargs
        param = sig.parameters[name]
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL, # *args
            inspect.Parameter.VAR_KEYWORD # **kwargs
        ): continue

        # append to original_kwargs
        orig_kwargs[name] = value

    return orig_kwargs

def cloneable(obj):
    '''
    (1) updates __init__ to capture and store original args/kwargs when initialized (_orig_kwargs)
    (2) adds clone() method to build new instance of the same type using _orig_kwargs
    (3) adds is_trained flag to nn.Module (models)
    '''
    # get original init and its signatuire (args, kwargs)
    orig_init = obj.__init__
    sig = inspect.signature(orig_init) 

    # define new init (orig init + kwargs capture)
    @functools.wraps(orig_init) # preserve name, doc, typehints, etc.
    def __init__(self, *args, **kwargs):

        # capture, store orig_kwargs on instance
        self._orig_kwargs = capture_kwargs(sig, self, *args, **kwargs)

        # store training status on instance
        if isinstance(self, nn.Module):
            self.is_trained = False

        # call original init
        orig_init(self, *args, **kwargs)

    # define clone func
    def clone(self, with_state:bool=False, **override_kwargs):
        # ensure _orig_kwargs is stored and exists
        if not hasattr(self, '_orig_kwargs'):
            raise RuntimeError(f'{type(self).__name__} was not initialized via @cloneable, _orig_kwargs is missing.')

        # merge orig kwargs with override kwargs (if applicable)
        init_kwargs = {**self._orig_kwargs, **override_kwargs}

        # build clone
        new_obj = type(self)(**init_kwargs)

        # load state_dict (if applicable)
        if with_state and hasattr(self, 'state_dict') and hasattr(new_obj, 'load_state_dict'):
            new_obj.load_state_dict(self.state_dict())

            if hasattr(self, 'is_trained'): 
                new_obj.is_trained = self.is_trained

        elif hasattr(new_obj, 'is_trained'):
            new_obj.is_trained = False

        return new_obj
    
    # add new init and clone func to obj
    obj.__init__ = __init__
    obj.clone = clone
    return obj

def get_layers(
    in_channels:int, 
    out_channels:int, 
    layer_class:nn.Module, 
    hidden_dims:list[int]=None, 
    act_fn:nn.Module=None, 
    norm_fn:Literal['batch','layer']=None, 
    end_fn:Union[bool,nn.Module]=False,
    layer_kwargs:dict={}
):
    '''
    * dynamically constructs a ModuleList of a given layer_class and act_fn. 
    * first two args of the layer_class must be (in_channels, out_channels).
    * act_fn can be module class or module (pre-init)
    * end_fn=True uses act_fn, or can use separate nn.Module; False no final.
    '''
    # init
    layers = nn.ModuleList()
    in_dim = in_channels # first in_dim is in_channels

    # defaults
    hidden_dims = [] if hidden_dims is None else hidden_dims
    act_fn = nn.ReLU if act_fn is None else act_fn
    
    # helper
    def add_fn(layers:nn.ModuleList, item:Union[nn.Module,tuple]):
        if isinstance(item, type) and issubclass(item, nn.Module): # initialize and append
            item = item()
            item.forward = filter_kwargs(item.forward)
            layers.append(item)
        elif isinstance(item, nn.Module): # deepcopy pre-initialized
            item = copy.deepcopy(item)
            item.forward = filter_kwargs(item.forward)
            layers.append(copy.deepcopy(item))
        else:
            raise TypeError(f'unsupported type: {type(item)}')

    # define hidden layers
    for hidden_dim in hidden_dims:
        # init class
        layer = layer_class(in_dim, hidden_dim, **layer_kwargs)
        layer.forward = filter_kwargs(layer.forward) # filter kwargs in forward
        layers.append(layer)

        # norm
        if norm_fn == 'batch':
            add_fn(layers, nn.BatchNorm1d(hidden_dim))
        elif norm_fn == 'layer':
            add_fn(layers, nn.LayerNorm(hidden_dim))

        # activation function
        add_fn(layers, act_fn)

        # set next in_dim as current hidden_dim
        in_dim = hidden_dim 

    # final layer
    layer = layer_class(in_dim, out_channels, **layer_kwargs)
    layer.forward = filter_kwargs(layer.forward) # filter kwargs in forward
    layers.append(layer)

    # end fn
    if end_fn is True:
        add_fn(layers, act_fn) # true = use act_fn
    elif end_fn is not False:
        add_fn(layers, end_fn) # custom end_fn

    return layers

def attn_dims(embed_dim:Optional[int]=None, head_dim:Optional[int]=None, num_heads:int=1):
    # none specified; assert error
    assert (embed_dim is not None) or (head_dim is not None), 'one of [embed_dim, head_dim] must be specified'

    # both specified; lin_out reshapes head to embed
    if (embed_dim is not None) and (head_dim is not None):
        assert embed_dim // num_heads == head_dim, 'transformer dims incompatible, (embed_dim // num_heads == head_dim) must be true'
        return embed_dim, head_dim, num_heads

    # embed_dim specified; head = embed / num_heads
    elif embed_dim is not None:
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        head_dim = embed_dim // num_heads

    # head_dim specified; embed = head * num_heads
    elif head_dim is not None:
        embed_dim = head_dim * num_heads

    return embed_dim, head_dim, num_heads

# data
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

        # print
        if self.verbose:
            for gpu in self.gpu_info:
                print(gpu)
            print()

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

        # set device
        torch.set_default_device(device)

        # print
        vprint(f'# device = {torch.get_default_device().__str__()}\n', verbose=self.verbose)
        return device
    
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
        device = self.set_device(device)

        return device