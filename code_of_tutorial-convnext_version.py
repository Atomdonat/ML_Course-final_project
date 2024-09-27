from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import copy

label_map = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}

label_count = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0
}

def finetuning_torchvision_models():
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

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

                    label_count[labels] += 1

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
    input_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0931, 0.0648, 0.0542])  # ImagNET default: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0931, 0.0648, 0.0542])  # ImagNET default: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ]),
    }

    print("\x1b[32m\nInitializing Datasets and Dataloaders...\033[38;5;188m")

    # Create training and validation datasets
    eurosat_dataset = datasets.eurosat.EuroSAT('./', download=True, transform=data_transforms['train'])
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

    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=1e-3, weight_decay=5e-4)  # optim.SGD(params_to_update, lr=0.001, momentum=0.9), optim.Adam(model_ft.parameters(), lr=learning_rate)
    sc_optimizer_ft = optim.AdamW(model_ft.parameters(), lr=5e-4, weight_decay=5e-4)  # optim.SGD(params_to_update, lr=0.001, momentum=0.9), optim.Adam(model_ft.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    sc_criterion_ft = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, from_scratch=False)
    _,scratch_hist = train_model(scratch_model, dataloaders_dict, sc_criterion_ft, sc_optimizer_ft, num_epochs=num_epochs, from_scratch=True)


if __name__ == '__main__':
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./eurosat"

    # Number of classes in the dataset
    num_classes = 10

    # Batch size for training (change depending on how much memory you have)
    # Todo: increase while memory allows it, for better performance
    batch_size = 32  # 160: 21/24 GB or 87.5% VRAM usage

    # Number of epochs to train for
    num_epochs = 5

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    current_model = finetuning_torchvision_models()

    for i in label_count:
        print(i)