import torch
import subprocess

def quickstart_tip():
    print(
'''*** quickstart_tip() ***
device, generator = u.Devices().auto_set_device()
brca = d.Data(
    counts_filepath='./data/dataFilt.csv', 
    sampletype_filepath='./data/sampletype_subtype.csv',
    relation_filepath='./data/relation_ohe.csv',
    drop = ['Normal', 'Metastatic'],
    max_subset = 120,
    verbose=verbose
)
'''
    )

def vprint(*objects, verbose=True, **kwargs):
    if verbose==True:
        print(*objects, **kwargs)

def preview_tensor(tensor:torch.tensor, tensor_name:str = '', print_tensor = True):
    # set default tensor name, if not provided
    if tensor_name == '':
        tensor_name = 'tensor shape'

    # print tensor shape
    print(f'{tensor_name}: {tensor.shape}')

    # print/return tensor
    if print_tensor == True:
        print(tensor, '\n')
    return tensor

def preview_model(model:torch.nn.Module, tensor:torch.tensor, model_name:str='model', print_tensor=False):
    # print model
    print(model, '\n')
    
    # print X, model(X)
    preview_tensor(tensor, 'X', print_tensor=print_tensor)
    preview_tensor(model(tensor), f'{model_name}(X)', print_tensor=print_tensor)

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
        vprint('*** Device() ***', verbose=self.verbose)

        # set device, generator
        torch.set_default_device(device)
        generator = torch.Generator(device=device)

        # print
        vprint(f'device = {torch.get_default_device().__str__()}\n', verbose=self.verbose)
        return device, generator
    
    def auto_set_device(self):
        # check device to use, cuda > mps > cpu
        if torch.cuda.is_available():
            device = self.gpu_list[0] # use gpu with highest free memory
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        # set device
        device, generator = self.set_device(device)

        return device, generator