import copy
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface

from .datasets import WindowDataset


logger = logging.getLogger(__name__)


class PyTorchModelTrainer(PyTorchTrainerInterface):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        data_convertor: PyTorchDataConvertor,
        model_meta_data: dict[str, Any] | None = None,
        window_size: int = 1,
        tb_logger: Any = None,
        **kwargs,
    ):
        """
        :param model: The PyTorch model to be trained.
        :param optimizer: The optimizer to use for training.
        :param criterion: The loss function to use for training.
        :param device: The device to use for training (e.g. 'cpu', 'cuda').
        :param init_model: A dictionary containing the initial model/optimizer
            state_dict and model_meta_data saved by self.save() method.
        :param model_meta_data: Additional metadata about the model (optional).
        :param data_convertor: converter from pd.DataFrame to torch.tensor.
        :param n_steps: used to calculate n_epochs. The number of training iterations to run.
            iteration here refers to the number of times optimizer.step() is called.
            ignored if n_epochs is set.
        :param n_epochs: The maximum number batches to use for evaluation.
        :param batch_size: The size of the batches to use during training.
        """
        if model_meta_data is None:
            model_meta_data = {}
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_meta_data = model_meta_data
        self.device = device
        self.n_epochs: int | None = kwargs.get("n_epochs", 10)
        self.n_steps: int | None = kwargs.get("n_steps", None)
        if self.n_steps is None and not self.n_epochs:
            raise Exception("Either `n_steps` or `n_epochs` should be set.")

        self.batch_size: int = kwargs.get("batch_size", 64)
        self.data_convertor = data_convertor
        self.window_size: int = window_size
        self.tb_logger = tb_logger
        self.test_batch_counter = 0

        # Gradient clipping: max L2 norm of gradients, 0 disables clipping
        self.max_grad_norm: float = kwargs.get("max_grad_norm", 1.0)

        # LR scheduler: cosine annealing from lr down to eta_min over all epochs
        self.lr_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None

        # Early stopping parameters
        self.early_stopping_patience: int = kwargs.get("early_stopping_patience", 0)
        self.best_val_loss: float = float("inf")
        self.patience_counter: int = 0
        self.best_model_state: dict | None = None

    def fit(self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]):
        """
        :param data_dictionary: the dictionary constructed by DataHandler to hold
        all the training and test data/labels.
        :param splits: splits to use in training, splits must contain "train",
        optional "test" could be added by setting freqai.data_split_parameters.test_size > 0
        in the config file.

         - Calculates the predicted output for the batch using the PyTorch model.
         - Calculates the loss between the predicted and actual output using a loss function.
         - Computes the gradients of the loss with respect to the model's parameters using
           backpropagation.
         - Updates the model's parameters using an optimizer.
        """
        self.model.train()

        # Enable TF32 on Ampere/Hopper/Blackwell GPUs (RTX 3090+) for faster matmul
        # with negligible precision loss. No-op on CPU or older GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary["train_features"])
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)

        # Cosine annealing decays LR smoothly from optimizer's lr down to eta_min
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6
        )

        batch_counter = 0
        for epoch in range(n_epochs):
            for _, batch_data in enumerate(data_loaders_dictionary["train"]):
                xb, yb = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                loss = self.criterion(yb_pred, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.tb_logger.log_scalar("train_loss", loss.item(), batch_counter)
                batch_counter += 1

            self.lr_scheduler.step()
            self.tb_logger.log_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            # evaluation
            if "test" in splits:
                val_loss = self.estimate_loss(data_loaders_dictionary, "test")

                if val_loss is not None:
                    improved = val_loss < self.best_val_loss
                    if improved:
                        self.best_val_loss = val_loss
                        self.best_model_state = copy.deepcopy(self.model.state_dict())

                    # Early stopping check
                    if self.early_stopping_patience > 0:
                        if improved:
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.early_stopping_patience:
                                logger.info(
                                    f"Early stopping triggered after {self.patience_counter} "
                                    f"epochs without improvement. "
                                    f"Best val_loss: {self.best_val_loss:.6f}"
                                )
                                break

        # Restore the best weights seen during training
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model weights (val_loss: {self.best_val_loss:.6f})")

    @torch.no_grad()
    def estimate_loss(
        self,
        data_loader_dictionary: dict[str, DataLoader],
        split: str,
    ) -> float | None:
        """
        Estimate loss on a data split.

        :param data_loader_dictionary: dictionary of data loaders.
        :param split: split to estimate loss on (e.g. "test").
        :return: average loss over all batches, or None if no batches.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for _, batch_data in enumerate(data_loader_dictionary[split]):
            xb, yb = batch_data
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)
            total_loss += loss.item()
            num_batches += 1
            self.tb_logger.log_scalar(f"{split}_loss", loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1

        self.model.train()

        if num_batches > 0:
            return total_loss / num_batches
        return None

    def create_data_loaders_dictionary(
        self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
        """
        data_loader_dictionary = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset = TensorDataset(x, y)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary

    def calc_n_epochs(self, n_obs: int) -> int:
        """
        Calculates the number of epochs required to reach the maximum number
        of iterations specified in the model training parameters.

        the motivation here is that `n_steps` is easier to optimize and keep stable,
        across different n_obs - the number of data points.
        """
        if not isinstance(self.n_steps, int):
            raise ValueError("Either `n_steps` or `n_epochs` should be set.")
        n_batches = n_obs // self.batch_size
        n_epochs = max(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(
                f"Setting low n_epochs: {n_epochs}. "
                f"Please consider increasing `n_steps` hyper-parameter."
            )

        return n_epochs

    def save(self, path: Path):
        """
        - Saving any nn.Module state_dict
        - Saving model_meta_data, this dict should contain any additional data that the
          user needs to store. e.g. class_names for classification models.
        """

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_meta_data": self.model_meta_data,
                "pytrainer": self,
            },
            path,
        )

    def load_from_checkpoint(self, checkpoint: dict):
        """
        when using continual_learning, DataDrawer will load the dictionary
        (containing state dicts and model_meta_data) by calling torch.load(path).
        you can access this dict from any class that inherits IFreqaiModel by calling
        the get_init_model method.
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_meta_data = checkpoint["model_meta_data"]
        return self


class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    Creating a trainer for the Transformer model.
    """

    def create_data_loaders_dictionary(
        self, data_dictionary: dict[str, pd.DataFrame], splits: list[str]
    ) -> dict[str, DataLoader]:
        """
        Converts the input data to PyTorch tensors using a data loader.
        """
        data_loader_dictionary = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f"{split}_features"], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f"{split}_labels"], self.device)
            dataset = WindowDataset(x, y, self.window_size)
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0,
            )
            data_loader_dictionary[split] = data_loader

        return data_loader_dictionary
