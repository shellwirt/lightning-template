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

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.datasets import fetch_california_housing
from pl_bolts.datamodules import SklearnDataModule
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError, R2Score


class LitNeuralModel(pl.LightningModule):
    """PyTorch Lightning's subclass for processing the model.

    NOTE: Please consult the documentation before performing modifications
    on the classes that this class overrides.
    """

    def __init__(self, learning_rate=1e-2, batch_size=32):
        """Initializes the Model with required parameters."""

        # Performs initialization
        super().__init__()

        # Saves learning rate as a internal parameter of the class
        self.learning_rate = learning_rate

        # Saves batch size as a internal parameter of the class
        self.batch_size = batch_size

        # Define metrics that will be logged
        metrics = MetricCollection([MeanAbsoluteError(), MeanSquaredError(), R2Score()])

        # Define training metrics with prefix
        self.train_metrics = metrics.clone(prefix="train_")

        # Define validation metrics with prefix
        self.valid_metrics = metrics.clone(prefix="val_")

        # Define test metrics with prefix
        self.test_metrics = metrics.clone(prefix="test_")

        # Defines sequential architecture
        self.learning_model = nn.Sequential(nn.Linear(8, 1))

        # Saves all hyperparameters to be logged by internal logging functionality
        self.save_hyperparameters(ignore=["learning_model"])

    # Forward method that runs the model
    def forward(self, X):

        # Feeds model to return the prediction
        out = self.neural(X)

        # Returns the prediction
        return out

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
        """Initialize optimizer and scheduler that will be used for training the model."""

        # Initialize LBFGS optimizer
        optimizer = torch.optim.LBFGS(
            self.learning_model.parameters(), lr=self.learning_rate
        )

        # Initialize ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        # Returns optimizer and scheduler configuration to the Trainer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "internal_valid_loss",
            },
        }


# Load features and labels from Sklearn's California Housing Dataset
X, y = fetch_california_housing(return_X_y=True)

# Perform reshape of labels so that they are not a 1-dimensional vector
# but a n-dimensional tensor with 1 label per tensor
y = y.reshape(len(y), 1)

# NOTE: This section contains code that defines the reproducibility of the model
# and is not meant to be changed unless the model expresses erractic behavior.

# Defines the RNG seed to be used by all the libraries
seed_everything(0)

# Set PyTorch's to only use deterministic algorithms
torch.use_deterministic_algorithms(True)

# Define the Early Stopping method for stopping trainings
early_stopping = EarlyStopping(monitor="internal_valid_loss", mode="min", patience=10)

# Define the model checkpoint callback for testing with multiple models
model_checkpoint = ModelCheckpoint(
    monitor="internal_valid_loss", mode="min", save_top_k=10, save_last=True
)

# Initialize trainer for initial learning rate tuner and batch size finder
trainer = pl.Trainer(
    deterministic=True,
    log_every_n_steps=1,
    max_epochs=-1,
    callbacks=[early_stopping, model_checkpoint],
)

# Initialize tuner for finding the optimal batch size
batchSuggestion = trainer.tuner.scale_batch_size(
    LitNeuralModel(), SklearnDataModule(X, y)
)

# Initialize tuner for finding the optimal learning rate
learningRate = trainer.tuner.lr_find(
    LitNeuralModel(), SklearnDataModule(X, y, batch_size=batchSuggestion)
)

# Initialize model with suggested parameters
neuralModel = LitNeuralModel(
    learning_rate=learningRate.suggestion(), batch_size=batchSuggestion
)

# Initialize data module with suggested parameters
dataModule = SklearnDataModule(X, y, batch_size=batchSuggestion)

# Tune the Neural Model with the optimal initial learning rate and batch size
trainer.tune(neuralModel, dataModule)

# Fits the model with the training and validation data
trainer.fit(neuralModel, dataModule)

# Load the best model checkpoint with the holdout test data
trainer.test(
    model=neuralModel.load_from_checkpoint(model_checkpoint.best_model_path),
    datamodule=dataModule,
)
