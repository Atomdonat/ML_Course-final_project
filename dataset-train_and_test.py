from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
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
    """Imshow for Tensor."""
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


def create_train_and_test_datasets(image_dir: str, train_size: float) -> (torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    return torch.utils.data.random_split(dataset, [train_size, 1-train_size])


if __name__ == '__main__':
    # data_dir = './EuroSAT_Images/EuroSAT_RGB'
    # X_train, X_test = create_train_and_test_datasets(data_dir, 0.8)
    #
    # train_data_loader = torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True)
    # test_data_loader = torch.utils.data.DataLoader(X_test, batch_size=32, shuffle=True)
    #
    # # Run this to test your data loader
    # images, labels = next(iter(train_data_loader))
    # imshow(images[0])
    # plt.show()

    esat = datasets.eurosat.EuroSAT('./', download=True)