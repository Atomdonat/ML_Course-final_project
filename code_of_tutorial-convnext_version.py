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


def finetuning_torchvision_models(learning_rate):
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    
    pretrained_trainging_accuracy_history = []
    pretrained_validation_accuracy_history = []
    pretrained_trainging_loss_history = []
    pretrained_validation_loss_history = []
    scratch_trainging_accuracy_history = []
    scratch_validation_accuracy_history = []
    scratch_trainging_loss_history = []
    scratch_validation_loss_history = []

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, from_scratch: bool = False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        if not from_scratch:
            print("\x1b[32m\nPre-trained weights loaded, training from default\n\033[38;5;188m")
        else:
            print("\x1b[32m\nNo weights loaded, training from scratch\n\033[38;5;188m")

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
                if not from_scratch:
                    if phase == 'train':
                            pretrained_trainging_accuracy_history.append(float(epoch_acc))
                            pretrained_trainging_loss_history.append(float(epoch_loss))
                    elif phase == 'val':
                        pretrained_validation_accuracy_history.append(float(epoch_acc))
                        pretrained_validation_loss_history.append(float(epoch_loss))
                else:
                    if phase == 'train':
                        scratch_trainging_accuracy_history.append(float(epoch_acc))
                        scratch_trainging_loss_history.append(float(epoch_loss))
                    elif phase == 'val':
                        scratch_validation_accuracy_history.append(float(epoch_acc))
                        scratch_validation_loss_history.append(float(epoch_loss))
                    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('\x1b[34mTraining complete in {:.0f}m {:.0f}s\033[38;5;188m'.format(time_elapsed // 60, time_elapsed % 60))
        print('\x1b[34mBest val Acc: {:4f}\033[38;5;188m'.format(best_acc))

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

        if use_pretrained:
            model_ft = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            model_ft = models.convnext_tiny(weights=None)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = 3 # Hardcoded in torchvision.models.convnenxt: layers.append(Conv2dNormActivation(...))
        # model_ft.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model_ft.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)
        input_size = 224

        return model_ft, input_size

    # Initialize the model for this run
    _, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

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

    print("\x1b[32m\nInitializing Datasets and Dataloaders...\033[38;5;188m")

    # Create training and validation datasets
    eurosat_dataset = datasets.ImageFolder('./eurosat/2750')
    train_dataset, validation_dataset = torch.utils.data.random_split(eurosat_dataset, [0.8, 0.2])

    train_dataset.dataset.transform = data_transforms['train']
    validation_dataset.dataset.transform = data_transforms['val']

    image_datasets = {'train': train_dataset, 'val': validation_dataset}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft,_ = initialize_model(num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)

    scratch_model, _ = initialize_model(num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)

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

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)  # optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    sc_optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)  # optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    sc_criterion_ft = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)  # optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, from_scratch=False)
    _,scratch_hist = train_model(scratch_model, dataloaders_dict, sc_criterion_ft, sc_optimizer_ft, num_epochs=num_epochs, from_scratch=True)

    return (
        pretrained_trainging_accuracy_history, 
        pretrained_trainging_loss_history,
        pretrained_validation_accuracy_history,
        pretrained_validation_loss_history, 
        scratch_trainging_accuracy_history, 
        scratch_trainging_loss_history, 
        scratch_validation_accuracy_history, 
        scratch_validation_loss_history
    )


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
    num_epochs = 2

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    current_model = finetuning_torchvision_models(
            learning_rate=0.005,
    )

    ptah, ptlh, pvah, pvlh, stah, stlh, svah, svlh = current_model
    
    plot_training(
        pretrained_accuracy_history=ptah,
        pretrained_loss_history=ptlh,
        scratch_accuracy_history=stah,
        scratch_loss_history=stlh,
    )

    # hist = {
    #     "pretrained_trainging_accuracy_history_list": [
    #         0.9251388888888888,
    #         0.9616203703703703,
    #         0.96875,
    #         0.9718055555555555,
    #         0.9737962962962963
    #     ],
    #     "pretrained_validation_accuracy_history_list": [
    #         0.9551851851851851,
    #         0.9594444444444444,
    #         0.9653703703703703,
    #         0.9653703703703703,
    #         0.9653703703703703
    #     ],
    #     "pretrained_trainging_loss_history_list": [
    #         0.2484274262079486,
    #         0.11589555773470137,
    #         0.0942358837083534,
    #         0.08386080654131042,
    #         0.07655131601625019
    #     ],
    #     "pretrained_validation_loss_history_list": [
    #         0.12922823583638227,
    #         0.11539074916530538,
    #         0.10539069711058228,
    #         0.10570607389564868,
    #         0.10262144000993835
    #     ],
    #     "scratch_trainging_accuracy_history_list": [
    #         0.12342592592592593,
    #         0.12342592592592593,
    #         0.12342592592592593,
    #         0.12342592592592593,
    #         0.12342592592592593
    #     ],
    #     "scratch_validation_accuracy_history_list": [
    #         0.12796296296296295,
    #         0.12796296296296295,
    #         0.12796296296296295,
    #         0.12796296296296295,
    #         0.12796296296296295
    #     ],
    #     "scratch_trainging_loss_history_list": [
    #         2.409748919804891,
    #         2.4097488862496834,
    #         2.409748930401272,
    #         2.409748923337018,
    #         2.4097488950800012
    #     ],
    #     "scratch_validation_loss_history_list": [
    #         2.4094337534021446,
    #         2.4094338364071315,
    #         2.4094337940216066,
    #         2.4094338399392585,
    #         2.409433756934272
    #     ]
    # }

    # plot_training(
    #     pretrained_accuracy_history=hist["pretrained_trainging_accuracy_history_list"],
    #     pretrained_loss_history=hist["pretrained_trainging_loss_history_list"],
    #     scratch_accuracy_history=hist["scratch_trainging_accuracy_history_list"],
    #     scratch_loss_history=hist["scratch_trainging_loss_history_list"]
    # )
