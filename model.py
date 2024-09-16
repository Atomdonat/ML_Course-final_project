from torchvision.models.convnext import ConvNeXt, convnext_tiny
import torch.optim as optim
import os
import torch
import psutil
import platform
import cpuinfo
from numba import cuda
from numba.cuda.cudadrv import enums


class Device:
    def __init__(self):
        self._current_device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._current_device

    @device.setter
    def device(self, new_device: str):
        self._current_device = torch.device(new_device)

    @staticmethod
    def available_devices() -> list[str]:
        """
        Returns a list of available devices in the format of 'cuda:index (device name)' or 'cpu' if no CUDA supporting GPU is found
        """
        _devices = []

        # Check for GPU availability
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                _devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
        else:
            _devices.append("cuda:0 (No CUDA device available)")

        # CPU is always available
        _devices.append("cpu")

        return _devices

    @staticmethod
    def get_gpu_info() -> None:
        device = cuda.get_current_device()
        compute_capability = f"{device.COMPUTE_CAPABILITY_MAJOR}.{device.COMPUTE_CAPABILITY_MINOR}"
        sm_count = device.MULTIPROCESSOR_COUNT  # Number of Streaming Multiprocessors (SM)

        # Mapping from Compute Capability to CUDA Cores per SM
        cuda_cores_per_sm = {
            "3": 192,  # Kepler
            "5": 128,  # Maxwell
            "6": 64,  # Pascal
            "7": 64,  # Volta, Turing
            "8": 128,  # Ampere
        }

        # Get number of cores per SM based on major version of Compute Capability
        major_version = str(device.COMPUTE_CAPABILITY_MAJOR)
        cores_per_sm = cuda_cores_per_sm.get(major_version, "Unknown architecture")

        if cores_per_sm == "Unknown architecture":
            return f"Unsupported Compute Capability: {compute_capability}"

        total_cuda_cores = sm_count * cores_per_sm
        cuda_core_info = {
            "GPU Name": torch.cuda.get_device_name(0),
            'Total Memory (MB)': torch.cuda.get_device_properties(0).total_memory // (1024 ** 2),
            "Total CUDA Cores": total_cuda_cores,
            # "SM Count": sm_count,
            # "CUDA Cores per SM": cores_per_sm,
            'PyTorch CUDA Version': torch.version.cuda,
            # 'Number of CUDA GPUs': torch.cuda.device_count(),
            'Compute Capability': f"{device.COMPUTE_CAPABILITY_MAJOR}.{device.COMPUTE_CAPABILITY_MINOR}"
        }

        print("\nGPU Information:")
        for key, value in cuda_core_info.items():
            print(f"{key}: {value}")


    @staticmethod
    def get_cpu_info() -> None:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)

        _cpu_info = cpuinfo.get_cpu_info()
        _cpu_info['cores'] = physical_cores
        _cpu_info['threads'] = logical_cores

        cpu_info = {
            'CPU Name': _cpu_info['brand_raw'],
            'Arch': _cpu_info['arch'],
            'Physical Cores': _cpu_info['cores'],
            'Logical Cores': _cpu_info['threads'],
            'Frequency': _cpu_info['hz_actual_friendly'],
            'L2 Cache': str(_cpu_info['l2_cache_size'] / 1024 ** 2) + ' MiB',
            'L3 Cache': str(_cpu_info['l3_cache_size'] / 1024 ** 2) + ' MiB',
        }

        print("\nCPU Information:")
        for key, value in cpu_info.items():
            print(f"{key}: {value}")


# Load entire model object
def load_current_model(path: str):
        """
        Load the entire model object and set it to evaluation mode.

        :References:
        - Load Entire Model: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model

        :param path: Path to the saved model file (File Extension should be `.pt` or `.pth`)
        :return: The loaded model instance
        """
        if not os.path.exists(path):
            raise ValueError(f"Path '{path}' does not exist")

        # Load the entire model (architecture + weights)
        loaded_model = torch.load(path)

        # Check if the loaded model is of the correct type
        if not isinstance(loaded_model, ConvNeXtTinyEuroSAT):
            raise ValueError("Loaded model does not match the current model class")

        # Set the model to evaluation mode and return
        loaded_model.eval()
        return loaded_model


# Load entire Model as TorchScript Model
def import_torch_script_model(path: str):
    """
    Load a TorchScript model from a file. Since TorchScript models are not typical nn.Module models,
    this method will load the entire model as TorchScript and overwrite the current instance.

    :References:
    - Import TorchScript: https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format

    :param path: Path to the TorchScript model file (File Extension should be `.pt` or `.pth`)
    :return: The loaded TorchScript model
    """
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")

    loaded_model = torch.jit.load(path)

    if not isinstance(loaded_model, torch.jit.ScriptModule):
        raise ValueError("Loaded model is not a TorchScript model")

    return loaded_model


class ConvNeXtTinyEuroSAT(ConvNeXt):
    """
    ConvNeXt Image Classification Model in the Tiny variant for the analysis of EuroSAT
    Explanation of ConvNext Implementation in ChatGPT.md -> Defining a Model

    :References:
    - ConvNeXt Tiny: https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html
    """
    def __init__(self, prior_weights,**kwargs):
        """
        :References:
        - ConvNeXt Tiny Weights: https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.ConvNeXt_Tiny_Weights

        :param prior_weights: The pretrained weights to use. See ConvNeXt_Tiny_Weights for more details and possible values. By default, no pre-trained weights are used.
        :param kwargs: Additional arguments passed to ConvNeXt:
                       block_setting (List[CNBlockConfig]);
                       stochastic_depth_prob (float);
                       layer_scale (float);
                       num_classes (int);
                       block (Optional[Callable[..., nn.Module]]);
                       norm_layer (Optional[Callable[..., nn.Module]]);
        """
        super(ConvNeXtTinyEuroSAT, self).__init__()
        self.model = convnext_tiny(
            weights=prior_weights,

        )
        self._optimizer = None
        self.epoch: int = 0
        self.loss: float = 0.  # Todo: is it a float?

    # _method_name avoids collisions with superclass methods
    def _train(self):
        # ToDo: implement me
        self.to(self.device)
        pass

    def _test(self):
        # ToDo: implement me
        pass

    def _predict(self):
        # ToDo: implement me
        pass

    @property
    def optimizer(self) -> optim.Optimizer:
        """
        :return: `torch.optim.Optimizer` object
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_class, **kwargs):
        """
        Set new Optimizer for the model or change Optimizer Parameters

        :References:
        - PyTorch Optimizer: https://pytorch.org/docs/stable/optim.html#algorithms

        :param optimizer_class: optimizer class from torch.optim module (e.g., torch.optim.SGD, torch.optim.Adam)
        :param kwargs: keyword arguments for the optimizer, reference optimizer documentation for more details
        """

        if not callable(optimizer_class):
            raise ValueError("Optimizer must be a callable class from torch.optim (e.g., torch.optim.SGD, torch.optim.Adam)")

        self._optimizer = optimizer_class(self.parameters(), **kwargs)

    # Load and Save state_dict
    def load_current_state_dict(self, path: str, _device: torch.device = None):
        """
        Load the model's parameters (weights and biases) in the form of a state dictionary (state_dict),
        which is a Python dictionary mapping each layer to its corresponding parameter tensor.

        :References:
        - Load Model State Dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        :param path: Path to the saved Model (File Extension should be `.pt` or `.pth`)
        :param _device: Optional argument to load model to a specific device (CPU or GPU)
        """
        if not os.path.exists(path):
            raise ValueError(f"Path '{path}' does not exist")

        # Load the state dict
        state_dict = torch.load(path, map_location=_device)

        # Load the state dict into the model
        self.load_state_dict(state_dict)
        self.eval()  # Set the model to evaluation mode

    def save_current_state_dict(self, path: str):
        """
        Save the model's parameters (weights and biases) in the form of a state dictionary (state_dict),
        which is a Python dictionary mapping each layer to its corresponding parameter tensor.

        :References:
        - Save Model State Dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        :param path: Path to save the current Model (File Extension should be `.pt` or `.pth`)
        """
        # Ensure the directory exists
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the state dict (weights and biases)
        torch.save(self.state_dict(), path)

    def print_model_state_dict(self):
        """
        Prints the state dict of the model

        :References:
        - Model State Dict: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
        """
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def print_optimizer_state_dict(self):
        """
        Prints the optimizer state dict of the model

        :References:
        - Optimizer State Dict: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict
        """
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])

    # Save the entire model object
    def save_current_model(self, path: str):
        """
        Save the entire model object, including the architecture, parameters, and possibly other information
        (e.g., the optimizer, buffers, etc.) using Python's pickle module.

        :References:
        - Pickle: https://docs.python.org/3/library/pickle.html
        - Save Entire Model: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model

        :param path: Path to save the current Model (File Extension should be `.pt` or `.pth`)
        """
        # Ensure the directory exists
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the entire model
        torch.save(self, path)

    # Save entire Model as TorchScript Model
    def export_as_torch_script_model(self, path: str):
        """
        Export the model as TorchScript.

        :References:
        - Export TorchScript: https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format

        :param path: Path to save the TorchScript model (File Extension should be `.pt` or `.pth`)
        """
        # Ensure the directory exists
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Export the model to TorchScript
        model_scripted = torch.jit.script(self)
        model_scripted.save(path)

    # Save and Load for training
    def save_current_state(self, path: str):
        """
        Saving a General Checkpoint for Inference

        :References:
        - Saving current State: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save

        :param path: Path to save the current Model TorchScript (File Extension should be `.tar`)
        """
        if not os.path.exists(path):
            raise ValueError(f"Path '{path}' does not exist")

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.state_dict(),
            'loss': self.loss,
        }, path)

    def load_current_state(self, path: str, is_eval: bool = True):
        """
        Loading a General Checkpoint for Resuming Training

        :References:
        - Loading current State: https://pytorch.org/tutorials/beginner/saving_loading_models.html#load

        :param path: Path to load the current Model TorchScript (File Extension should be `.tar`)
        :param is_eval: If set to `True` the model is set to evaluation mode, otherwise it is set to training mode
        """
        if not os.path.exists(path):
            raise ValueError(f"Path '{path}' does not exist")

        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

        if is_eval:
            self.eval()
        else:
            self.train()


if __name__ == "__main__":
    # TORCH_SCRIPT_MODEL_PATH = './'
    devices = Device()
    print(devices.available_devices())

    devices.get_cpu_info()
    devices.get_gpu_info()
    # devices.device = 'cuda:0'
