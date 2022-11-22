[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fshellwirt%2Flightning-template.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fshellwirt%2Flightning-template?ref=badge_shield)

# lightning-template
This repository contains a template that can be used to create new models using the Lightning Framework.

## Template Features
- Early Stopping Policy.
- Model Checkpoint saving top ten trained models based on holdout data.
- Learning Rate Finder and Batch Size Finder with automatic saving for model training.
- MSE, MAE, R2 metrics saved to local Tensorboard for training and holdout data.
- ReduceLROnPlateau as learning rate scheduler and LBFGS optimizer.
- Reproducibility Policy with deterministic algorithms only.

## Requirements

For installing the required packages with **pip**, please run the following command:

```
pip install -r pip-requirements.txt
```

For installing the required packages with **conda**, please run the following command:

```
conda install --file conda-requirements.txt
```

## Running the template

Since this template is fully functional, there's no prerequisites for this code to run. Simply run the following command:

```
python template.py
```


## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fshellwirt%2Flightning-template.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fshellwirt%2Flightning-template?ref=badge_large)
