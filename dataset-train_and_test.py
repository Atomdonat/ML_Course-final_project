import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np



def plot_training(pretrained_accuracy_history, pretrained_loss_history, scratch_accuracy_history, scratch_loss_history):
    num_epochs = len(pretrained_accuracy_history) - 1
    _x = range(0, num_epochs + 1)

    # Plot Trendlines for each Accuracy
    if len(pretrained_accuracy_history) > 1:
        trend_pah = np.polyfit(_x, pretrained_accuracy_history, 2)
        poly_trend_pah = np.poly1d(trend_pah)
    else:
        poly_trend_pah = [0] * num_epochs

    if len(pretrained_loss_history) > 1:
        trend_plh = np.polyfit(_x, pretrained_loss_history, 2)
        poly_trend_plh = np.poly1d(trend_plh)
    else:
        poly_trend_plh = [0] * num_epochs

    if len(scratch_accuracy_history) > 1:
        trend_sah = np.polyfit(_x, scratch_accuracy_history, 2)
        poly_trend_sah = np.poly1d(trend_sah)
    else:
        poly_trend_sah = [0] * num_epochs

    if len(scratch_loss_history) > 1:
        trend_slh = np.polyfit(_x, scratch_loss_history, 2)
        poly_trend_slh = np.poly1d(trend_slh)
    else:
        poly_trend_slh = [0] * num_epochs

    plt.title("Training Accuracy and Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy and Loss")

    # Accuracy plot
    plt.plot(_x, poly_trend_pah(_x), label="Pretrained Acc Trend", color="cyan", linewidth=1)
    plt.plot(_x, pretrained_accuracy_history, label="Pretrained Acc", color='blue', linewidth=1, linestyle="--")

    plt.plot(_x, poly_trend_sah(_x), label="Scratch Acc Trend", color="orange", linewidth=1)
    plt.plot(_x, scratch_accuracy_history, label="Scratch Acc", color='red', linewidth=1, linestyle="--")

    # Loss plot
    plt.plot(_x, poly_trend_plh(_x), label="Pretrained Loss Trend", color="cyan", linewidth=1)
    plt.plot(_x, pretrained_loss_history, label="Pretrained Loss", color='blue', linewidth=1, linestyle="--")

    plt.plot(_x, poly_trend_slh(_x), label="Scratch Loss Trend", color="orange", linewidth=1)
    plt.plot(_x, scratch_loss_history, label="Scratch Loss", color='red', linewidth=1, linestyle="--")

    # Y-axis limits
    plt.ylim((0, 1.))
    plt.yticks(np.arange(0, 1., 0.05))
    plt.xticks(np.arange(0, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()

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
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    try:
        dataset = datasets.eurosat.EuroSAT('./', download=True, transform=data_transforms['train'])

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading EuroSAT dataset: {e}")
        print("Falling back to ImageFolder...")
        dataset = datasets.ImageFolder('./eurosat/2750', transform=data_transforms['train'])

    train_dataset, validation_dataset = random_split(dataset, [train_size, 1 - train_size])

    return train_dataset, validation_dataset


if __name__ == '__main__':
    from model import *

    X_train, X_test = create_train_and_test_datasets(0.8)
    batch_size = 160  # 160: 21/24 GB or 87.5% VRAM usage

    train_data_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)

    devices = Device()
    devices.device = 'cuda:0'

    num_classes = 10
    num_epochs = 1

    cn_model = ConvNeXtTinyEuroSAT(
        device=devices.device
    )

    cn_model.optimizer = torch.optim.Adam(cn_model.parameters(), lr=1e-3)
    cn_model.criterion = torch.nn.CrossEntropyLoss()

    cn_model.train_and_validate_model(
        training_dataloader=train_data_loader,
        validation_dataloader=test_data_loader,
        num_epochs=num_epochs,
        pretrained_weights=convnext.ConvNeXt_Tiny_Weights
    )