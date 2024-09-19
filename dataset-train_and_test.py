from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
# from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def imshow(image, ax=None, title=None, normalize=True):
    """
    Display an image tensor using Matplotlib.

    This function takes a tensor image, optionally normalizes it, and displays it using Matplotlib.
    It supports the optional display of the image on a specified Axes object. If no Axes object is provided,
    a new figure and Axes are created.

    Args:
        image (torch.Tensor): The image tensor to be displayed. Expected shape is (C, H, W), where C is the number
                              of channels (3 for RGB images), H is height, and W is width.
        ax (matplotlib.axes.Axes, optional): The Axes object on which to display the image. If None, a new figure and
                                             Axes are created. Default is None.
        title (str, optional): Title to set for the image. If provided, it will be set on the Axes. Default is None.
        normalize (bool, optional): Whether to normalize the image using mean and standard deviation values
                                    (common for pre-trained models). Default is True.

    Returns:
        matplotlib.axes.Axes: The Axes object with the displayed image. If a new Axes object was created, it will
                               be returned; otherwise, the provided Axes object is returned.

    Notes:
        The normalization is performed using the following mean and standard deviation values:
        - Mean: [0.485, 0.456, 0.406]
        - Std: [0.229, 0.224, 0.225]

        The function also hides the spines and tick marks for a cleaner display of the image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def create_train_and_test_datasets(train_size: float) -> (torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset):
    """
    Create train and test datasets from the specified directory.
    Uses EuroSAT dataset if available; otherwise, falls back to ImageFolder (should be stored like './EuroSAT_Images/EuroSAT_RGB').

    :param train_size: Proportion of the dataset to use for training (between 0 and 1).
    :return: A tuple containing the training and testing datasets.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    try:
        dataset = datasets.eurosat.EuroSAT('./', download=True, transform=transform)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading EuroSAT dataset: {e}")
        print("Falling back to ImageFolder...")
        dataset = datasets.ImageFolder('./eurosat/2750', transform=transform)

    train_dataset, test_dataset = random_split(dataset, [train_size, 1-train_size])

    return train_dataset, test_dataset


if __name__ == '__main__':
    X_train, X_test = create_train_and_test_datasets(0.8)

    train_data_loader = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=True)

    # Run this to test your data loader
    images, labels = next(iter(train_data_loader))
    imshow(images[0])
    plt.show()
