# PDE-NHF

Official implementation of PDE-NHF, a Hamiltonian-based Normalizing Flows models for integrating Vlasov-Poisson equations.
Right now, the code supports 1D experiments.


## Setup

Install the following packages, using the pip command in a terminal with Python 3.10 or 3.11:

- torch

- numpy

- random

- argparse

- matplotlib


## Folders

- **[data](data/)**: This is where the training data should be saved.

- **[models](model/)**: This is where the models should be saved, each one in a different folder.



## Generating and saving datasets (Step 1)

This is the first step for training and/or testing a model.
All datasets should be saved in a folder using a .npy extension.
A dataset can be created using the [generate_data.py](generate_data.py) file. To create a dataset, run

```commandline
python generate_data.py -SEED=1 -N_EXAMPLES=32768 -N_PARTICLES=256 -MIN_STD_Q=0.5 -MAX_STD_Q=1.5 -MIN_STD_P=0.5 -MAX_STD_P=1.5
```

which will create the dataset in the data folder. 

```
optional arguments:
  -h, --help            show this help message and exit.

  -SEED                 Random seed.
  
  -FOLDER_DATA          Folder in which data are saved.

  -N_EXAMPLES           Number of examples to generate.

  -N_PARTICLES          Number of particles on the grid per example.

  -MIN_STD_Q            Minimum of initial positions standard deviation.
  
  -MAX_STD_Q            Maximum of initial positions standard deviation.
  
  -MIN_STD_P            Minimum of initial momenta standard deviation.
  
  -MIN_STD_P            Maximum of initial momenta standard deviation.
```

Initial, intermediate and final positions and momenta are saved after training as well as 
info regarding the standard deviations of the initial phase space distributions. Note that
initial positions and momenta are assumed to be centered.



## Training a model (Step 2)

The [train_model.py](train_model.py) script takes care of performing the training.
To start training, run

```commandline
python train_model.py -SEED=1 -FOLDER_EXP='models/model/' -FOLDER_DATA='data/' -N_TRAINING=20000 -N_VALIDATION=6384 -L=25 -DT=0.04 -N_EPOCHS=200 -BATCH_SIZE=128 -LR=0.0003
```

```
optional arguments:
  -h, --help            show this help message and exit.

  -SEED                 Random seed.
  
  -FOLDER_EXP           Folder in which model is saved.
  
  -FOLDER_DATA          Folder in which data are saved.

  -N_TRAINING           Number of training examples.

  -N_VALIDATION         Number of validation examples.

  -L                    Number of Leapfrog steps.

  -DT                   Timestep in Leapfrog scheme.

  -N_EPOCHS             Number of training epochs.

  -BATCH_SIZE           Minibatch size during training.

  -LR                   Learning rate for Adam optimizer.
```

`model_final`, `optimizer_final`, `training_loss_final` and `validation_loss_final` are saved after training. 
Right now, there is a saved trained model in the subfolder called [models/model/](models/model/). 
Data generation (Step 1) with the above options is needed to test it.



## Post-process (Step 3)

You can assess the performance of the model with the Jupyter Notebook [post_process.ipynb](post_process.ipynb).




