# ConvNeXt Image Classification for EuroSAT LULC

## Installation
1) Install [PyTorch](https://pytorch.org/get-started/locally/#start-locally) with prepared command in Python Environment (CUDA/GPU only on Nvidia Graphic-Cards)
2) Install other Python/Conda Packages: matplotlib, Pillow/PIL, (pycuda, py-cpuinfo)

## Training results
### Pretrained Parameter Tuning
#### narrowing down learning rates
- weight decay of 1e-2 (default):
![](training_history/plots/pretrained_finetuning/pretrained_learnrate_wd_1e_2_tuning.png)
- weight decay of 5e-3:
![](training_history/plots/pretrained_finetuning/pretrained_learnrate_wd_5e_3_tuning.png)
**&rarr; tuning Parameter for learn rates 1e-4, 2.5e-4, 5e-4** (red, orange, green)
#### narrowing down weight decays per learn rate
- learn rate of 1e-4:
![](training_history/plots/pretrained_finetuning/constant_learning_rate/learning_rate_1e_4.png)
- learn rate of 2.5e-4:
![](training_history/plots/pretrained_finetuning/constant_learning_rate/learning_rate_25e_5.png)
- learn rate of 5e-4:
![](training_history/plots/pretrained_finetuning/constant_learning_rate/learning_rate_5e_4.png)
**&rarr; we are going forward with lr=5e-4, wd=**
## Troubleshooting
- ~~the original Dataset link is unusable due to an invalid SSL CERT which won't get renewed or replaced (old link is hardcoded in `torchvision.datasets.eurosat.EuroSAT()` ¯\(°_o)/¯)~~
  - ~~instead of using `datasets.EuroSAT()` we are using `datasets.ImageFolder()` (EuroSAT is subclass of ImageFolder)~~
  - somehow `torchvision.datasets.eurosat.EuroSAT()` works again (thanks, i guess?)
- OMP Error:
```text
OMP: Error \#15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```
  - reinstalling numpy

## Glossary
- **ConvNeXt: The ConvNeXt model is based on the [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) paper.**
- **EuroSAT:** Eurosat is a dataset and deep learning benchmark for land use and land cover classification. The dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.
- **CNN:** Convolutional Neural Network (https://paperswithcode.com/paper/eurosat-a-novel-dataset-and-deep-learning)
- **Overfitting:** If the training loss decreases significantly but the validation loss increases, it might indicate overfitting.
- **Underfitting:** If both training and validation loss remain high, it might indicate underfitting.

### Accuracy:
- Definition: The percentage of correct predictions made by the model on a given dataset.
- Interpretation: A higher accuracy indicates that the model is making more correct predictions. However, it's important to consider the context of the problem. For example, an accuracy of 90% might be excellent for one task but insufficient for another.
- Usage: Often used to assess overall performance, especially in classification tasks.

### Loss:
- Definition: A measure of how far the model's predictions are from the true values.
- Interpretation: A lower loss value indicates that the model's predictions are closer to the true values. Different loss functions are used for different types of tasks (e.g., mean squared error for regression, cross-entropy loss for classification).
- Usage: Loss is often used to guide the model's training process. The goal is to minimize the loss.

### Interpreting Overfitting:
- With Loss values:
  - Training loss: Decreases steadily as the model learns.
  - Validation loss: Should also decrease initially.
  - Overfitting: If the validation loss starts to increase while the training loss continues to decrease, it's a strong indicator of overfitting. The model is becoming too specialized to the training data and struggles to generalize to unseen data.
- With Accuracy values: 
  - Training accuracy: Increases steadily as the model learns. 
  - Validation accuracy: Should also increase initially.
  - Overfitting: If the validation accuracy starts to decrease or plateau while the training accuracy continues to increase, it's another sign of overfitting. The model is memorizing the training data rather than learning underlying patterns.

## ToDo:
- [ ] Random Seed for Reproducibility
- [ ] Cross-Validation 
- [x] Parameter Sweeping (Hyperparameter Tuning)
- [ ] create boxplot

## Bibliography
- [EuroSAT](https://github.com/phelber/EuroSAT)
  - [EuroSAT Files](https://zenodo.org/api/records/7711810/files-archive)
- [PyTorch](https://pytorch.org/)
  - [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
  - [ConvNeXt Tiny](https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html)
  - [Optimizer](https://pytorch.org/docs/stable/optim.html#algorithms) 
  - [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## Comment Highlighting with macro's: 
- TODO:
  - Usage: Used to mark tasks that need to be done.
  - Pattern: `\btodo\b*`
  - Main Color (Light Green): `#A4C639`
  - Darker Version: `#8CAF2D`
- FIXME:
  - Usage: Used to highlight code that needs fixing.
  - Pattern: `\bfixme\b*`
  - Main Color (Light Red or Pink): `#FF6F61`
  - Darker Version: `#E65B4F`
- BUG:
  - Usage: Used to mark known bugs in the code.
  - Pattern: `\bbug\b*`
  - Main Color (Light Red or Pink): `#FF6F61`
  - Darker Version: `#E65B4F`
- IDEA:
  - Usage: Used to denote an idea or suggestion for the code.
  - Pattern: `\bidea\b*`
  - Main Color (Light Yellow): `#FFD700`
  - Darker Version: `#E6BE00`
- NOTE:
  - Usage: Used to add notes or explanations about the code.
  - Pattern: `\bnote\b*`
  - Main Color (Light Blue): `#ADD8E6`
  - Darker Version: `#93C2CF`
- WARNING:
  - Usage: Used to indicate something that might need attention or could be problematic.
  - Pattern: `\bwarning\b*`
  - Main Color (Orange): `#FFA500`
  - Darker Version: `#E69500`
- HACK:
  - Usage: Used to mark code that is a workaround or temporary solution.
  - Pattern: `\bhack\b*`
  - Main Color (Purple): `#9370DB`
  - Darker Version: `#7D60BF`