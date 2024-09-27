import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
import random



def move_random_files(source_dir, destination_dir, count):
    files = os.listdir(source_dir)

    while count > 0:
        file_to_move = random.choice(files)
        sd = os.path.join(source_dir, file_to_move)
        dd = os.path.join(destination_dir, file_to_move)
        shutil.move(sd, dd)

        count -= 1
        files.pop(files.index(file_to_move))


def create_val_dirs(source_dir):
    for i in os.listdir(source_dir):
        vpth = os.path.join(source_dir, i)
        if not os.path.exists(vpth):
            os.makedirs(vpth)


def create_eurosat_dataloader(train_size: float = 0.8, input_size: int = 224, batch_size: int = 32) -> dict[torch.utils.data.dataset.Subset]:
    """
    Create train and test datasets from the specified directory.
    Uses EuroSAT dataset if available; otherwise, falls back to ImageFolder (should be stored like './EuroSAT_Images/EuroSAT_RGB').

    :param train_size: Proportion of the dataset to use for training (between 0 and 1, default is 0.8).
    :param input_size: Size of the Crop for the images in transforms.Compose(). (Default is 224)
    :return: A tuple containing the training and testing datasets.
    """
    print("\x1b[32m\nInitializing Datasets and Dataloaders...\x1b[38;5;188m\n")

    if 0 >= train_size >= 1.:
        raise ValueError("Train size must be between 0 and 1.")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0931, 0.0648, 0.0542])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0931, 0.0648, 0.0542])
        ]),
    }

    try:
        eurosat_dataset = datasets.eurosat.EuroSAT('./', download=True, transform=data_transforms['train'])

        os.rename("./eurosat/2750", "./eurosat/train")



    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading EuroSAT dataset: {e}")
        print("Falling back to ImageFolder...")

        eurosat_dataset = datasets.ImageFolder('./eurosat/train', transform=data_transforms['train'])

    # finally:
        # train_dataset, val_dataset = random_split(eurosat_dataset, [train_size, 1-train_size])
        #
        # train_dataset.dataset.transform = data_transforms['train']
        # val_dataset.dataset.transform = data_transforms['val']
        # # Create training and validation dataloaders
        # dataloaders_dict = {
        #     'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        #     'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        # }
        #
        # return dataloaders_dict


if __name__ == '__main__':

    # X_train, X_test =\
    # create_eurosat_dataloader()
    batch_size = 160  # 160: 21/24 GB or 87.5% VRAM usage


    # train_data_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    # test_data_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)
    #
    # devices = Device()
    # devices.device = 'cuda:0'
    #
    # num_classes = 10
    # num_epochs = 1
    #
    # cn_model = ConvNeXtTinyEuroSAT(
    #     device=devices.device
    # )
    #
    # cn_model.optimizer = torch.optim.Adam(cn_model.parameters(), lr=1e-3)
    # cn_model.criterion = torch.nn.CrossEntropyLoss()
    #
    # cn_model.train_and_validate_model(
    #     training_dataloader=train_data_loader,
    #     validation_dataloader=test_data_loader,
    #     num_epochs=num_epochs,
    #     pretrained_weights=convnext.ConvNeXt_Tiny_Weights
    # )