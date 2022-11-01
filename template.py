""" 
BSD 3-Clause License

Copyright (c) 2022, shellwirt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import random

import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.datasets import fetch_california_housing
from pl_bolts.datamodules import SklearnDataModule
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError


class LitNeuralModel(pl.LightningModule):
    """PyTorch Lightning's subclass for processing the model.

    NOTE: Please consult the documentation before performing modifications
    on the classes that this class overrides.
    """

    def __init__(self, learning_model, learning_rate):
        """Initializes the Model with required parameters."""

        # Performs initialization
        super().__init__()

        # Saves learning rate as a internal parameter of the class
        self.learning_rate = learning_rate

        # Saves model as a internal parameter of the class
        self.learning_model = learning_model

        # Define metrics that will be logged
        metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError()])

        # Define training metrics with prefix
        self.train_metrics = metrics.clone(prefix="train_")

        # Define validation metrics with prefix
        self.valid_metrics = metrics.clone(prefix="val_")

        # Define test metrics with prefix
        self.test_metrics = metrics.clone(prefix="test_")

        # Saves all hyperparameters to be logged by internal logging functionality
        self.save_hyperparameters(ignore=["learning_model"])

    def training_step(self, batch, batch_idx):
        """Defines the training step to feed the model and calculate loss."""

        # Initialize features and labels with the informed batch size
        x, y = batch

        # Performs the prediction of the model
        y_hat = self.learning_model(x)

        # Calculate the loss for the optimizer and monitoring
        loss = F.mse_loss(y_hat, y)

        # Calculate the loss for all metrics to be logged
        output = self.train_metrics(y_hat, y)

        # Save calculated metrics to disk
        self.log_dict(output)

        # Save the loss metric for internal monitoring and debugging purposes
        self.log("internal_training_loss", loss)

        # NOTE: In this case, we are allowed to return the loss to the model to
        # the Trainer, since doing so we'll update the model state.
        return loss

    def test_step(self, batch, batch_idx):
        """Defines the test step to check for generalization."""

        # Initiates features and labels with the informed batch size
        x, y = batch

        # Performs the prediction of the model
        y_hat = self.learning_model(x)

        # Calculate the loss for logging the generalization of the model
        loss = F.mse_loss(y_hat, y)

        # Calculate the loss for all metrics to be logged
        output = self.test_metrics(y_hat, y)

        # Save calculated metrics to disk
        self.log_dict(output)

        # Save the loss metric for internal monitoring and debugging purposes
        self.log("internal_test_loss", loss)

    def validation_step(self, batch, batch_idx):
        """Defines the validation step for checking model convergence."""

        # Initializes features and labels with the informed batch size
        x, y = batch

        # Performs prediction of the model
        y_hat = self.learning_model(x)

        # Calculates the loss for logging the convergence of the model
        loss = F.mse_loss(y_hat, y)

        # Calculate the loss for all metrics to be logged
        output = self.valid_metrics(y_hat, y)

        # Save calculated metrics to disk
        self.log_dict(output)

        # Save the loss metric for internal monitoring and debugging purposes
        self.log("internal_valid_loss", loss)

    def configure_optimizers(self):
        """Initialize optimizers that will be used for training the model."""

        # Initialize Adam optimizer for automaticaly updating the learning rate
        optimizer = torch.optim.Adam(
            self.learning_model.parameters(), lr=self.learning_rate, weight_decay=2
        )

        # Initialize the Step scheduler for changing the learning rate throughout training.
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)

        # Returns optimizer and learning scheduler to the Trainer
        return [optimizer], [lr_scheduler]


class NeuralNetwork(nn.Module):
    """Linear Regression with Sequential Model.

    A fully connected sequential model with Linear Regression.

    NOTE: Before running the model, it is required to validate the input dimension
    for proper training.

    """

    # Initialize the class with the Sequential Model
    def __init__(self):

        # Performs the class initialization
        super(NeuralNetwork, self).__init__()

        # Defines sequential architecture
        self.linear = nn.Sequential(nn.Linear(8, 1))

    # Forward method that runs the model
    def forward(self, X):

        # Feeds model to return the prediction
        out = self.linear(X)

        # Returns the prediction
        return out


# Load features and labels from Sklearn's California Housing Dataset
X, y = fetch_california_housing(return_X_y=True)

# Perform reshape of labels so that they are not a 1-dimensional vector
# but a n-dimensional tensor with 1 label per tensor
y = y.reshape(len(y), 1)

# Load features and labels in DataModule
loaders = SklearnDataModule(X, y, batch_size=500)

# Initialize training loader
train_loader = loaders.train_dataloader()

# Initialize validation loader
val_loader = loaders.val_dataloader()

# Initialize testing loader
test_loader = loaders.test_dataloader()

# NOTE: This section contains code that defines the reproducibility of the model
# and is not meant to be changed unless the model expresses erractic behavior.

# Defines the RNG seed to be used by all the libraries
manual_seed = 0

# Set Python's random seed to make sure that any core operator is deterministic
random.seed(manual_seed)

# Set NumPy's random seed to make sure that any library operator is deterministic
np.random.seed(manual_seed)

# Set Pytorch's to only use a manually defined seed for all devices
torch.manual_seed(manual_seed)

# Set PyTorch's to only use deterministic algorithms
torch.use_deterministic_algorithms(True)

# Initialize trainer for initial learning rate tuner
tune_trainer = pl.Trainer(auto_lr_find=True, log_every_n_steps=1, deterministic=True)

# Find the optimal initial learning rate for the model
lr_suggestion = tune_trainer.tuner.lr_find(
    LitNeuralModel(NeuralNetwork(), learning_rate=1e-2),
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

# Initialize the Neural Model with the optimal initial learning rate
neuralmodel = LitNeuralModel(
    learning_model=NeuralNetwork(), learning_rate=lr_suggestion.suggestion()
)

# Initialize the model trainer which feeds the model
trainer = pl.Trainer(
    log_every_n_steps=1,
    max_epochs=1000,
    deterministic=True,
    callbacks=[
        # Define the Early Stopping method for stopping trainings
        EarlyStopping(monitor="internal_valid_loss", mode="min", patience=5),
    ],
)

# Fits the model with the training and validation data
trainer.fit(model=neuralmodel, datamodule=loaders)

# Validates model generalization with the test data
trainer.test(neuralmodel, datamodule=loaders)
