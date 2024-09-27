from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import time
import copy
import inspect


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def train_convnext(dataloaders_dict, num_epochs: int = 5):
    def initialize_model(num_classes, feature_extract, use_pretrained=True):
        if use_pretrained:
            model_ft = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            model_ft = models.convnext_tiny(weights=None)

        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False

        model_ft.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)
        model_ft.to(device)
        return model_ft

    def train_model(model, dataloaders, lr, weight_decay, num_epochs: int = 5):
        since = time.time()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        val_acc_history = []

        current_parameter = inspect.cleandoc(f"""\
            criterion:    nn.CrossEntropyLoss 
            optimizer:    optim.AdamW
            lr:           {str(lr)}
            weight_decay: {str(weight_decay)}
            num_epochs:   {str(num_epochs)}\n
        """)
        print(current_parameter)

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

                    for label in labels.cpu().tolist():
                        label_count[label] += 1

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
        print('\x1b[34mTraining complete in {:.0f}m {:.0f}s\x1b[38;5;188m'.format(time_elapsed // 60, time_elapsed % 60))
        print('\x1b[34mBest val Acc: {:4f}\x1b[38;5;188m'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        # return model, val_acc_history
        return model

    # pretrained_model = initialize_model(
    #     num_classes=10,
    #     feature_extract=True,
    #     use_pretrained=True
    # )
    #
    # pretrained_model_trained = train_model(
    #     model=pretrained_model,
    #     dataloaders=dataloaders_dict,
    #     lr=1e-3,
    #     weight_decay=5e-4,
    #     num_epochs=num_epochs
    # )
    # save_model(pretrained_model_trained, 'PretrainedIsTheCrown')

    scratch_model = initialize_model(
        num_classes=10,
        feature_extract=False,
        use_pretrained=False
    )

    scratch_score_trained = train_model(
        model=scratch_model,
        dataloaders=dataloaders_dict,
        lr=5e-4,
        weight_decay=5e-4,
        num_epochs=num_epochs
    )
    save_model(scratch_score_trained, 'TheScratchMachine')


def save_model(model, nickname):
    path = f'./training_history/models/{nickname}.pth'
    path_sd = f'./training_history/models/{nickname}_state_dict.pth'

    torch.save(model, path)
    torch.save(model.state_dict(), path_sd)


if __name__ == '__main__':
    input_size = 224
    batch_size = 128  # Pretrained: 160 (maybe more); Scratch: 128

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

    # Create training and validation datasets
    eurosat_dataset = datasets.eurosat.EuroSAT('./', download=True, transform=data_transforms['train'])
    train_dataset, validation_dataset = random_split(eurosat_dataset, [0.8, 0.2])

    train_dataset.dataset.transform = data_transforms['train']
    validation_dataset.dataset.transform = data_transforms['val']

    image_datasets = {'train': train_dataset, 'val': validation_dataset}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    train_convnext(
        dataloaders_dict=dataloaders_dict,
        num_epochs=15
    )
