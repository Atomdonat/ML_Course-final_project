# Changes for current commit

[//]: # (## .gitignore)

## dataset-train_and_test.py
- load and split dataset (train and test) accordingly
- added method to display images of dataset (user-friendly purpose)

## model.py
- created subclass of ConvNeXt to implement our Code
- added 4 methods each to load and save different parts of the model (state_dict, entire model as .pt, entire model as TorchScript, current Progress)
- added "dynamic" optimizer swapability 
- method to get relevant CPU (and GPU) Specifications to select correct one for CUDA and PyTorch  

## README.md
- added