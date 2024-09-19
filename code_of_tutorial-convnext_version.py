from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json


def finetuning_torchvision_models():
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.

        # Todo: replace pretrained [dep] with weights (e.g. ConvNeXt_Tiny_Weights(), previously saved ones or None)
        model_ft = models.convnext_tiny(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = 3 # Hardcoded in torchvision.models.convnenxt: layers.append(Conv2dNormActivation(...))
        # model_ft.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model_ft.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)
        input_size = 224

        return model_ft, input_size

    # Initialize the model for this run
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    eurosat_dataset = datasets.ImageFolder('./eurosat/2750')
    train_dataset, test_dataset = torch.utils.data.random_split(eurosat_dataset, [0.8, 0.2])

    train_dataset.dataset.transform = data_transforms['train']
    test_dataset.dataset.transform = data_transforms['val']

    image_datasets = {'train': train_dataset, 'val': test_dataset}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)  # optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    # Initialize the non-pretrained version of the model used for this run
    scratch_model,_ = initialize_model(num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.Adam(scratch_model.parameters(), lr=0.001)  #optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs)

    ohist = []
    shist = []

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]

    johist = [h.cpu().numpy().tolist() for h in hist]
    jshist = [h.cpu().numpy().tolist() for h in scratch_hist]

    time_now = time.time()

    with open(f'./history.json', 'r') as f:
        data = json.load(f)
        data[time_now] = {
            "ohist": johist,
            "shist": jshist
        }

    with open(f'./history.json', 'w') as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    return ohist, shist


def plot_training(pretrained_accuracy_history, pretrained_loss_history, scratch_accuracy_history, scratch_loss_history):
    num_epochs = len(pretrained_accuracy_history) - 1
    _x = range(0, num_epochs + 1)

    # Plot Trendlines for each Accuracy
    trend_pah = np.polyfit(_x, pretrained_accuracy_history, 2)
    trend_plh = np.polyfit(_x, pretrained_loss_history, 2)
    trend_sah = np.polyfit(_x, scratch_accuracy_history, 2)
    trend_slh = np.polyfit(_x, scratch_loss_history, 2)

    poly_trend_pah = np.poly1d(trend_pah)
    poly_trend_plh = np.poly1d(trend_plh)
    poly_trend_sah = np.poly1d(trend_sah)
    poly_trend_slh = np.poly1d(trend_slh)

    plt.title("Training Accuracy and Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy and Loss")

    # Accuracy plot
    plt.plot(_x, pretrained_accuracy_history, label="Pretrained Acc", color='cyan', linewidth=1, linestyle="--")
    plt.plot(_x, scratch_accuracy_history, label="Scratch Acc", color='orange', linewidth=1, linestyle="--")
    plt.plot(_x, poly_trend_pah(_x), label="Pretrained Acc Trend", color="cyan", linewidth=1)
    plt.plot(_x, poly_trend_sah(_x), label="Scratch Acc Trend", color="red", linewidth=1)

    # Loss plot
    plt.plot(_x, pretrained_loss_history, label="Pretrained Loss", color='green', linewidth=1, linestyle="--")
    plt.plot(_x, scratch_loss_history, label="Scratch Loss", color='magenta', linewidth=1, linestyle="--")
    plt.plot(_x, poly_trend_plh(_x), label="Pretrained Loss Trend", color="cyan", linewidth=1)
    plt.plot(_x, poly_trend_slh(_x), label="Scratch Loss Trend", color="red", linewidth=1)

    # Y-axis limits
    plt.ylim((0, 1.))
    plt.yticks(np.arange(0, 1., 0.05))
    plt.xticks(np.arange(0, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./eurosat"

    # Number of classes in the dataset
    num_classes = 10

    # Batch size for training (change depending on how much memory you have)
    # Todo: increase while memory allows it, for better performance
    batch_size = 160  # 160: 21/24 GB or 87.5% VRAM usage

    # Number of epochs to train for
    num_epochs = 14

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # oh, sh = finetuning_torchvision_models()
    # plot_accuracies(pretrained_accuracy_history=oh, scratch_accuracy_history=sh)
    plot_training(
        pretrained_accuracy_history=[0.9493, 0.9561, 0.9635, 0.9652, 0.9654, 0.9665, 0.9672, 0.9659, 0.9656, 0.9681, 0.9687, 0.9681, 0.9693, 0.97, 0.9683],
        pretrained_loss_history=[0.1794, 0.1418, 0.1206, 0.1107, 0.1087, 0.1028, 0.1011, 0.0969, 0.0988, 0.0949, 0.0948, 0.094, 0.0944, 0.0912, 0.093],
        scratch_accuracy_history=[0.9493, 0.9561, 0.9635, 0.9652, 0.9654, 0.9665, 0.9672, 0.9659, 0.9656, 0.9681, 0.9687, 0.9681, 0.9693, 0.97, 0.9683],
        scratch_loss_history=[0.1794, 0.1418, 0.1206, 0.1107, 0.1087, 0.1028, 0.1011, 0.0969, 0.0988, 0.0949, 0.0948, 0.094, 0.0944, 0.0912, 0.093]
    )