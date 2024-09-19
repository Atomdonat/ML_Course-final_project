# ConvNeXt Image Classification for EuroSAT LULC

## Installation
1) Install [PyTorch](https://pytorch.org/get-started/locally/#start-locally) with prepared command in Python Environment (CUDA/GPU only on Nvidia Graphic-Cards)
2) Install other Python/Conda Packages: matplotlib, Pillow/PIL, (pycuda, py-cpuinfo)

## Troubleshooting
- ~~the original Dataset link is unusable due to an invalid SSL CERT which won't get renewed or replaced (old link is hardcoded in `torchvision.datasets.eurosat.EuroSAT()` ¯\(°_o)/¯)~~
  - ~~instead of using `datasets.EuroSAT()` we are using `datasets.ImageFolder()` (EuroSAT is subclass of ImageFolder)~~
  - somehow `torchvision.datasets.eurosat.EuroSAT()` works again (thanks, i guess?)
- OMP Error:
```text
OMP: Error \#15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```
  - [ ] reinstalling numpy
  
## Glossary
- **CNN:** Convolutional Neural Network
- **EuroSAT:** Collection of European Satellite Images for Land Use and Land Cover Classification with Sentinel-2

[//]: # (- **:**)

[//]: # (- **:**)

[//]: # (- **:**)

[//]: # (- **:**)

[//]: # (- **:**)

[//]: # (- **:**)

[//]: # (- **:**)

[//]: # (- **:**)

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