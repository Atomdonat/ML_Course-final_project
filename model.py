from torchvision.models import convnext
import torch.optim as optim
import os
import torch
import psutil
import platform
import cpuinfo
from numba import cuda
from numba.cuda.cudadrv import enums
import time
import copy
from typing import Optional
import inspect
import json


class Device:
    def __init__(self):
        self._current_device = torch.device("cpu")

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
# Todo: rename path to ... filename
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
# Todo: rename path to ... filename
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


class ConvNeXtTinyEuroSAT(convnext.ConvNeXt):
    """
    ConvNeXt Image Classification Model in the Tiny variant for the analysis of EuroSAT

    :References:
    - ConvNeXt Tiny: https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html

    :param device: The torch device to use. By default, CPU is used.
    """
    def __init__(
            self,
            device: torch.device = 'cpu'
    ):
        block_setting = [
            convnext.CNBlockConfig(96, 192, 3),
            convnext.CNBlockConfig(192, 384, 3),
            convnext.CNBlockConfig(384, 768, 9),
            convnext.CNBlockConfig(768, None, 3),
        ]
        super(ConvNeXtTinyEuroSAT, self).__init__(block_setting)

        self.model = convnext.convnext_tiny(weights=None)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.set_parameter_requires_grad(True)
        self.model.classifier[2] = torch.nn.Linear(in_features=768, out_features=self.num_classes)

        self.device = device
        self.num_classes = 10  # Number of LULC Classes / Subdirectories in EuroSAT
        self.input_size = 224
        self.ohist = []
        self.shist = []
        self.pretrained_accuracy_history = []
        self.pretrained_loss_history = []
        self.scratch_accuracy_history = []
        self.scratch_loss_history = []
        self.save_time = 0

    def update_model_weights(self, weights: convnext.ConvNeXt_Tiny_Weights | None = None):
        """
        Set new ConvNeXt Tiny Model weights

        :references:
        - ConvNeXt Tiny Weights: https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.ConvNeXt_Tiny_Weights

        :param weights: The pretrained weights to use. See :class:`torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            for more details and possible values. By default, no pre-trained weights are used.
        """

        model_ft = convnext.convnext_tiny(weights=weights)

        self.set_parameter_requires_grad(True)
        model_ft.classifier[2] = torch.nn.Linear(in_features=768, out_features=self.num_classes)

        self.save_model()
        self.model = model_ft
        self.model.to(self.device)

    def train_and_validate_model(
            self,
            training_dataloader: torch.utils.data.DataLoader,
            validation_dataloader: torch.utils.data.DataLoader,
            num_epochs: int = 10,
            pretrained_weights: convnext.ConvNeXt_Tiny_Weights = None,
            from_scratch: bool = False
    ) -> convnext.ConvNeXt:
        """
        Train and validate the model on the given dataloaders.

        :param training_dataloader: Dataloader containing training data
        :type training_dataloader: torch.utils.data.DataLoader
        :param validation_dataloader: Dataloader containing validation data
        :type validation_dataloader: torch.utils.data.DataLoader
        :param num_epochs: Number of epochs to train the model
        :param pretrained_weights: The pretrained weights to use. See :class:`torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            for more details and possible values.
        :param from_scratch: Whether to train the model from scratch. By default, no pre-trained weights are used.

        :returns: Model with the best accuracy and updates self.ohist and self.shist
        """

        dataloaders = {'train': training_dataloader, 'val': validation_dataloader}

        def train_model():
            """
            Code adaptation of https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/finetuning_torchvision_models_tutorial.ipynb
            """
            since = time.time()

            val_acc_history = []
            pretrained_accuracy_history = []
            pretrained_loss_history = []
            scratch_accuracy_history = []
            scratch_loss_history = []

            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_acc = 0.0

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    if not from_scratch:
                        pretrained_accuracy_history.append(epoch_acc)
                        pretrained_loss_history.append(epoch_loss)
                    else:
                        scratch_accuracy_history.append(epoch_acc)
                        scratch_loss_history.append(epoch_loss)

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    if phase == 'val':
                        val_acc_history.append(epoch_acc)

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            self.model.load_state_dict(best_model_wts)
            return self.model, val_acc_history

        params_to_update = self.model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)

        if not from_scratch:
            self.update_model_weights(weights=pretrained_weights)
            self.model, hist = train_model()
            self.ohist = [h.cpu().numpy() for h in hist]

        else:
            self.update_model_weights(weights=None)
            _, scratch_hist = train_model()
            self.shist = [h.cpu().numpy() for h in scratch_hist]

        self.save_model()

    def save_model(self):
        # save Model
        self.save_time = time.strftime('%Y-%m-%d_%H-%M-%S')

        self.save_history_lists()
        self.save_current_model()
        self.save_current_state_dict()
        # self.export_as_torch_script_model()

    def update_optimizer(self, optimizer_class_instance: torch.optim.Optimizer):
        """
        Set new Optimizer for the model or change Optimizer Parameters

        :References:
        - PyTorch Optimizer: https://pytorch.org/docs/stable/optim.html#algorithms

        :param optimizer_class_instance: instantiated optimizer class from torch.optim module (e.g., torch.optim.SGD, torch.optim.Adam)
        """
        self.optimizer = optimizer_class_instance

    def update_criterion(self, criterion_instance: torch.nn.Module):
        """
        Set new Loss Function for the model

        :References:
        - PyTorch Loss Functions: https://pytorch.org/docs/stable/nn.html#loss-functions

        :param criterion_instance: instantiated loss function class from torch.nn module (e.g., torch.nn.CrossEntropyLoss, torch.nn.L1Loss)
        """
        self.criterion = criterion_instance

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    # Load and Save state_dict
    # Todo: rename path to ... filename
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

    def save_current_state_dict(self):
        """
        Save the model's parameters (weights and biases) in the form of a state dictionary (state_dict),
        which is a Python dictionary mapping each layer to its corresponding parameter tensor.

        :References:
        - Save Model State Dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
        """
        # Construct the path
        path = f'./training_history/state_dicts/{self.save_time}.pth'
        directory = os.path.dirname(path)

        # Ensure the directory exists
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
        print(f"{self.save_time} - Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def print_optimizer_state_dict(self):
        """
        Prints the optimizer state dict of the model

        :References:
        - Optimizer State Dict: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict
        """
        print(f"{self.save_time} - Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])

    # Save the entire model object
    def save_current_model(self):
        """
        Save the entire model object, including the architecture, parameters, and possibly other information
        (e.g., the optimizer, buffers, etc.) using Python's pickle module.

        :References:
        - Pickle: https://docs.python.org/3/library/pickle.html
        - Save Entire Model: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
        """
        # Construct the path
        path = f'./training_history/models/{self.save_time}.pth'
        directory = os.path.dirname(path)

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the entire model
        torch.save(self, path)

    # Save entire Model as TorchScript Model
    def export_as_torch_script_model(self):
        """
        Export the model as TorchScript.

        :References:
        - Export TorchScript: https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format

        """
        # Construct the path
        path = f'./training_history/TorchScript_models/{self.save_time}.pth'
        directory = os.path.dirname(path)

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Export the model to TorchScript
        # Fixme: throws error on execution
        model_scripted = torch.jit.script(self)
        model_scripted.save(path)

    # Save and Load for training
    def save_current_state(self):
        """
        Saving a General Checkpoint for Inference

        :References:
        - Saving current State: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
        """
        # Construct the path
        path = f'./training_history/checkpoint/{self.save_time}.pth'
        directory = os.path.dirname(path)

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.state_dict(),
            'loss': self.loss,
        }, path)

    # Todo: rename path to ... filename
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

    def save_history_lists(self):
        # convert lists and save stats f.e. to plot later
        pretrained_accuracy_history_list = [h.cpu().numpy().tolist() for h in self.pretrained_accuracy_history]
        pretrained_loss_history_list = [h.cpu().numpy().tolist() for h in self.pretrained_loss_history]
        scratch_accuracy_history_list = [h.cpu().numpy().tolist() for h in self.scratch_accuracy_history]
        scratch_loss_history_list = [h.cpu().numpy().tolist() for h in self.scratch_loss_history]

        with open(f'training_history/history.json', 'r') as f:
            data = json.load(f)
            data[self.save_time] = {
                "pretrained_accuracy_history_list": pretrained_accuracy_history_list,
                "pretrained_loss_history_list": pretrained_loss_history_list,
                "scratch_accuracy_history_list": scratch_accuracy_history_list,
                "scratch_loss_history_list": scratch_loss_history_list,
            }

        with open(f'training_history/history.json', 'w') as f:
            json.dump(data, f, indent=4)
            f.write("\n")

if __name__ == "__main__":
    devices = Device()
    devices.device = 'cuda:0'

    num_classes = 10
    num_epochs = 5

    convnext_weights_pretrained = convnext.ConvNeXt_Tiny_Weights.IMAGENET1K_V1  # Default pretrained weights
    convnext_weights_scratch = None

    cn_model = ConvNeXtTinyEuroSAT(device=devices.device)

    cn_model.update_model_weights(weights=None)
    cn_model.update_optimizer(torch.optim.Adam(cn_model.parameters(), lr=1e-3))
    cn_model.update_criterion(torch.nn.CrossEntropyLoss())

    for train_status in range(2):
        for current_lr in [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e-0]:

