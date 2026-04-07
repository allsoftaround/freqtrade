from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchTransformerTrainer
from freqtrade.freqai.torch.PyTorchTransformerModel import PyTorchTransformerModel


class PyTorchTransformerRegressor(BasePyTorchRegressor):
    """
    Transformer-based regressor for FreqAI using multi-head self-attention over a
    sliding window of past candles. Uses PyTorchTransformerModel internally.

    Parameters are passed via ``model_training_parameters`` in the freqai config block.

    Config example::

        "freqai": {
            "conv_width": 30,
            "feature_parameters": {
                "include_shifted_candles": 0
            },
            "model_training_parameters": {
                "learning_rate": 3e-4,
                "trainer_kwargs": {
                    "n_steps": 5000,
                    "batch_size": 64,
                    "n_epochs": null,
                    "early_stopping_patience": 10,
                    "max_grad_norm": 1.0,
                    "huber_delta": 1.0
                },
                "model_kwargs": {
                    "hidden_dim": 256,
                    "dropout_percent": 0.1,
                    "n_layer": 2,
                    "nhead": 4
                }
            }
        }

    model_kwargs:
        - hidden_dim (int): Internal model dimension. Must be divisible by nhead (auto-adjusted
          downward with a warning if not). Default: 256
        - n_layer (int): Number of Transformer encoder layers. Default: 2
        - dropout_percent (float): Dropout in encoder and output head. Default: 0.1
        - nhead (int): Number of self-attention heads. Default: 8

    trainer_kwargs:
        - n_steps (int): Training steps (used to derive n_epochs when n_epochs is null).
        - n_epochs (int | null): Explicit epoch count. Set null to use n_steps instead.
        - batch_size (int): Mini-batch size. Default: 64
        - early_stopping_patience (int): Stop if val_loss does not improve for this many
          epochs. 0 disables early stopping. Default: 0
        - max_grad_norm (float): Gradient clipping L2 norm. 0 disables. Default: 1.0
        - huber_delta (float): Delta parameter for HuberLoss. Default: 1.0
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs", {})

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        n_features = data_dictionary["train_features"].shape[-1]
        n_labels = data_dictionary["train_labels"].shape[-1]
        model = PyTorchTransformerModel(
            input_dim=n_features,
            output_dim=n_labels,
            time_window=self.window_size,
            **self.model_kwargs,
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        # HuberLoss (SmoothL1) is more robust than MSE for noisy financial data:
        # it uses MSE for small errors and MAE for large outliers, reducing the
        # impact of extreme price spikes on training.
        criterion = torch.nn.HuberLoss(delta=self.trainer_kwargs.get("huber_delta", 1.0))
        # check if continual_learning is activated, and retrieve the model to continue training
        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchTransformerTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                window_size=self.window_size,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )
        trainer.fit(data_dictionary, self.splits)
        return trainer

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[pd.DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        x = self.data_convertor.convert_x(
            dk.data_dictionary["prediction_features"], device=self.device
        )
        # if user is asking for multiple predictions, slide the window
        # along the tensor
        x = x.unsqueeze(0)
        # create empty torch tensor
        self.model.model.eval()
        yb = torch.empty(0).to(self.device)
        if x.shape[1] > self.window_size:
            ws = self.window_size
            for i in range(0, x.shape[1] - ws):
                xb = x[:, i : i + ws, :].to(self.device)
                y = self.model.model(xb)
                yb = torch.cat((yb, y), dim=1)
        else:
            yb = self.model.model(x)

        yb = yb.cpu().squeeze(0)
        pred_df = pd.DataFrame(yb.detach().numpy(), columns=dk.label_list)
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)

        if self.ft_params.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        if x.shape[1] > 1:
            zeros_df = pd.DataFrame(
                np.zeros((x.shape[1] - len(pred_df), len(pred_df.columns))), columns=pred_df.columns
            )
            pred_df = pd.concat([zeros_df, pred_df], axis=0, ignore_index=True)
        return (pred_df, dk.do_predict)
