import copy
import functools
import inspect
import itertools
import torch
import torch.nn as nn
import subprocess
import numpy as np
import pandas as pd

import math # stringify
import numbers # stringify
import re # clean_name
import unicodedata # clean_name
import hashlib # clean_name

# typing
from torch import Tensor
from torch_geometric.data import Batch, Data
from typing import Callable, Literal, Optional, Type, Union


# errors/sanity
def assert_finite(x:Tensor, name:str):
    if not torch.isfinite(x).all():
        num_bad = (~torch.isfinite(x)).sum().item()
        _min = x.min().item() if x.numel() else float('nan')
        _max = x.max().item() if x.numel() else float('nan')
        raise ValueError(f'{name} contains {num_bad} non-finite values (min={_min}, max={_max})')

def assert_nonnegative(x:Tensor, name:str):
    assert_finite(x, name)
    if (x < 0).any():
        _min = x.min().item()
        raise ValueError(f'{name} contains negative values (min={_min})')

# math
def tsoftmax(input:Tensor, temperature:float=1, dim:int=None, dtype:Optional[torch.dtype]=None):
    if temperature is None:
        temperature = 1
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    
    return torch.softmax(input / temperature, dim=dim)

# general
def stringify(x) -> str:
    # None, str, int, bool: as is
    if x is None:
        return 'None'
    if isinstance(x, str):
        return x
    if isinstance(x, (int,bool)):
        return str(x) # bool as 'True' or 'False'

    # float: sci not.
    if isinstance(x, numbers.Real):
        # float inf, -inf, nan
        if math.isinf(x) or math.isnan(x):
            return str(x)
        
        # sci not
        sci = f'{x:.2e}'
        mantissa, exp = sci.split('e')

        # short floats as reg
        if abs(float(exp)) < 3:
            return f'{x:.3g}'

        # long floats as formatted sci not.
        exp = exp.replace('-0','-')
        exp = exp.replace('+0','')
        mantissa = mantissa.rstrip('0').rstrip('.')
        return f'{mantissa}e{exp}'

    # type, callable: name or class name
    if isinstance(x, type) or callable(x):
        return getattr(x, "__name__", x.__class__.__name__)

    # containers: recursive
    if isinstance(x, list):
        return '[' + ','.join(stringify(i) for i in x) + ']'
    
    if isinstance(x, tuple):
        return '(' + ','.join(stringify(i) for i in x) + ')'
    
    if isinstance(x, set):
        # sort values compared as repr/str (deterministic), 
        items = sorted(x, key=lambda v: repr(v))
        return '{' + ','.join(stringify(i) for i in items) + '}'
    
    # dict: recursive
    if isinstance(x, dict):
        # sort (key,value) pairs by key compared as repr/str (deterministic), 
        items = sorted(x.items(), key=lambda kv: repr(kv[0]))
        return '{' + ','.join(stringify(k) + ':' + stringify(v) for k,v in items) + '}'

    # fallback, class name or force str
    try:
        return x.__class__.__name__
    except Exception:
        return str(x)
    
def clean_name(x, max_len:int=80) -> str:
    """
    Turn arbitrary Python object x into a filesystem-safe name
    for use as a single file or folder name on Linux, macOS, and Windows.
    """
    # stringify
    name = stringify(x)

    # normalize unicode (helps on macOS, etc.)
    name = unicodedata.normalize("NFKD", name)

    # remove path separators (just in case)
    name = name.replace("/", "_").replace("\\", "_")

    # collapse whitespace to single underscores
    name = re.sub(r"\s+", "_", name)

    # remove characters invalid on Windows:  < > : " / \ | ? *
    # (keep only: letters, digits, dash, underscore, dot, plus)
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)

    # collapse multiple underscores, dots
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"\.{2,}", ".", name)

    # trim leading/trailing spaces, dots, underscores
    name = name.strip(" ._")

    # avoid empty names
    if not name:
        name = "unnamed"

    # avoid Windows reserved device names (case-insensitive)
    upper = name.upper()
    windows_reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if upper in windows_reserved_names or name in {".", ".."}:
        name = name + "_"

    # truncate long names but keep a short hash suffix for uniqueness
    if len(name) > max_len:
        # generate hash suffix
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]

        # room for '_' + hash
        keep = max_len - 1 - len(digest)

        # extreme case: max_len very small, use digest only, up to max len
        if keep < 1:
            name = digest[:max_len]

        # otherwise, append digest suffix to name
        else:
            name = name[:keep] + "_" + digest

    return name

def get_name_recursive(obj):
    # type, callable, get __name__
    if isinstance(obj, type) or callable(obj):
        return getattr(obj, "__name__", obj.__class__.__name__)
    
    # dict, recurse through vals
    if isinstance(obj, dict):
        return {k: get_name_recursive(v) for k, v in obj.items()}
    
    # container, recurse
    if isinstance(obj, (list,tuple,set)):
        t = type(obj)
        return t(get_name_recursive(x) for x in obj)
    
    # keep real values for primitives
    if isinstance(obj, (str,int,bool,float)) or (obj is None):
        return obj
    
    # instances
    if isinstance(type(obj), type):
        return obj.__class__.__name__
    
    # fallback
    return stringify(obj)

# viz, pandas
def filter_df(df, filters={}):
    mask = pd.Series(True, index=df.index)

    for col, rule in filters.items():
        if callable(rule):
            mask &= rule(df[col])        # custom lambda
        elif isinstance(rule, (list, tuple, set)):
            mask &= df[col].isin(rule)
        else:
            mask &= (df[col] == rule)

    return df[mask]

def merge_devtest_csvs(folder, subfolders:list[str], file:str, save:bool=False):
    '''
    merges csvs with the same name in separate subfolders, with option to save to main folder
    useful for experiment main folder, and manually combining trials across subfolders.

    example structure:

    folder (e.g. experiment)
        subfolder1
            dev.csv
            test.csv
        subfolder2
            dev.csv
            test.csv
        subfolder3
            (etc.)

    folder should be str or Path
    subfolder is list of str
    file should be str

    '''
    trial_dfs = []
    old_max = -1 # start at -1+1=0

    for subfolder in subfolders:
        # read data
        trial_df = pd.read_csv(folder / subfolder / file)

        # safety, if file does not exist
        if trial_df.empty:
            continue

        # get new minimum (of current trial)
        new_min = trial_df['trial'].min()

        # calculate shift
        shift = (old_max+1) - new_min

        # shift trial to start after old max
        trial_df['trial'] += shift

        # append to list
        trial_dfs.append(trial_df)

        # get maximum of current trial for next (old_max)
        old_max = trial_df['trial'].max()

    df = pd.concat(trial_dfs, ignore_index=True)

    if save:
        df.to_csv(folder / file, index=False)

    return df

def validate_filter(values, all_values:list, name:str):
    # return all_values if None
    if values is None:
        return all_values

    # ensure values is list
    if not isinstance(values, list):
        values = [values]

    # check for filter vals missing in all vals
    missing = [x for x in values if x not in all_values]
    if missing:
        raise ValueError(f'Unknown {name}(s): {missing}')
    
    return values

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

def input_to_dict(x, name:str='x'):
    # already dict
    if isinstance(x, dict):
        return x
    
    # x (Tensor) only
    elif isinstance(x, Tensor): 
        return {name:x}

    # PyG Data or DataBatch
    elif isinstance(x, (Data, Batch)): 
        return dict(x)

    # error case
    else:
        raise TypeError(f'unsupported input type: {type(x)}')

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
def vprint(*objects, verbose:bool=True, **kwargs):
    if verbose:
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