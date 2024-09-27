# from dataset_train_and_test import *
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import copy
import inspect

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def optimize_convnext(dataloaders_dict, batch_size, input_size, scratch: bool = False, num_epochs: int = 5):
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

        if not scratch:
            print("\x1b[32m\nPre-trained weights loaded, training from default\n\x1b[38;5;188m")
        else:
            print("\x1b[32m\nNo weights loaded, training from scratch\n\x1b[38;5;188m")

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
        return best_acc

    def hyperparameter_tuning(param_grid: dict, dataloaders:dict, num_epochs: int = 5) -> (float, dict):
        best_score = 0
        best_params = None
        for params in ParameterGrid(param_grid):
            # Re-initialize model for unbiased results
            if not scratch:
                model = initialize_model(
                    num_classes=10,
                    feature_extract=True,
                    use_pretrained=True
                )
            else:
                model = initialize_model(
                    num_classes=10,
                    feature_extract=False,
                    use_pretrained=False
                )

            score = train_model(
                model=model,
                dataloaders=dataloaders,
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                num_epochs=num_epochs
            )

            torch.save(model, f"./{score:.4f}_nn-CrossEntropyLoss_optim-AdamW_{params['lr']}_{params['weight_decay']}.pth")

            if score > best_score:
                best_score = score
                best_params = params
                torch.save(model.state_dict(), f"./best_model_{best_score:.4f}_lr-{params['lr']}_wd-{params['weight_decay']}.pth")

        print("\x1b[34mBest Parameters: {}\x1b[38;5;188m".format(best_params))
        print("\x1b[34mBest Score: {}\x1b[38;5;188m".format(best_score))
        return best_score, best_params

    # Create a Grid of Hyperparameters
    param_grid = {
        # 'lr': [1e-4,2.5e-4,5e-4],  # for pretrained
        # 'lr': [5e-5, 8e-5, 1e-4, 2e-4, 2.5e-4, 5e-4],  # for scratch
        'lr': [1e-3],
        'weight_decay': [5e-4]  #[1e-5, 1e-4]
    }

    return hyperparameter_tuning(
        param_grid=param_grid,
        dataloaders=dataloaders_dict,
        num_epochs=num_epochs
    )


if __name__ == '__main__':
    batch_size = 32
    input_size = 224
    num_epochs = 15

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
    train_dataset, validation_dataset = torch.utils.data.random_split(eurosat_dataset, [0.8, 0.2])

    train_dataset.dataset.transform = data_transforms['train']
    validation_dataset.dataset.transform = data_transforms['val']

    image_datasets = {'train': train_dataset, 'val': validation_dataset}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    optimize_convnext(
        dataloaders_dict=dataloaders_dict,
        batch_size=batch_size,
        input_size=input_size,
        scratch=False,
        num_epochs=num_epochs
    )
