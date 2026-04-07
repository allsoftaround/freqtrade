import logging
import math

import torch
from torch import nn


logger = logging.getLogger(__name__)


"""
The architecture is based on the paper "Attention Is All You Need".
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017.
"""


class PyTorchTransformerModel(nn.Module):
    """
    A Transformer encoder model for time-series regression.

    Improvements over the original implementation:
    - Input features are projected to hidden_dim before the encoder, avoiding
      silent feature truncation caused by forcing input_dim to be divisible by nhead.
    - nhead is auto-adjusted downward to the largest value that divides hidden_dim,
      with a warning logged if the requested value is changed.
    - Pre-Norm (norm_first=True) on each encoder layer for more stable training.
    - Explicit dim_feedforward=4*hidden_dim (PyTorch default is 2048 regardless of hidden_dim).
    - Mean pooling over the time dimension replaces the previous flatten approach,
      keeping the output-head input size fixed at hidden_dim regardless of window size.
    - PositionalEncoding max_len is based on time_window, not feature dim (bug fix).

    :param input_dim: Number of input features per timestep.
    :param output_dim: Number of regression targets.
    :param hidden_dim: Internal model dimension (d_model). Default: 256
    :param n_layer: Number of Transformer encoder layers. Default: 2
    :param dropout_percent: Dropout rate applied in encoder and output head. Default: 0.1
    :param time_window: Number of past timesteps (candles) fed as a sequence. Default: 10
    :param nhead: Number of attention heads. Will be reduced if it does not divide
        hidden_dim. Default: 8

    :returns: Tensor of shape (batch_size, 1, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 1,
        hidden_dim: int = 256,
        n_layer: int = 2,
        dropout_percent: float = 0.1,
        time_window: int = 10,
        nhead: int = 4,
    ):
        super().__init__()
        self.time_window = time_window

        # Project raw features to hidden_dim so nhead only needs to divide hidden_dim,
        # not the raw feature count. This avoids silently dropping input features.
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_percent),
        )

        # Walk nhead down to the largest value that evenly divides hidden_dim
        valid_nhead = nhead
        while hidden_dim % valid_nhead != 0 and valid_nhead > 1:
            valid_nhead -= 1
        if valid_nhead != nhead:
            logger.warning(
                f"Requested nhead={nhead} does not divide hidden_dim={hidden_dim}. "
                f"Using nhead={valid_nhead} instead."
            )

        # max_len must cover the full time_window sequence length (the original code
        # incorrectly used dim_val which could be smaller than time_window)
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim, max_len=max(time_window * 2, 512)
        )

        # Pre-Norm encoder (norm_first=True) is more stable than Post-Norm.
        # dim_feedforward set explicitly to 4*hidden_dim (PyTorch default is 2048).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=valid_nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout_percent,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: nested tensor optimization is incompatible
        # with norm_first=True (Pre-Norm). Setting it explicitly silences the warning.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layer, enable_nested_tensor=False
        )

        # Mean-pool over the time dimension then project to output.
        # Input size is always hidden_dim regardless of time_window.
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor, mask=None, add_positional_encoding: bool = True):
        """
        :param x: Input tensor of shape [Batch, SeqLen, input_dim]
        :param mask: Optional attention mask passed to the encoder.
        :param add_positional_encoding: Whether to add positional encoding.
        :returns: Tensor of shape [Batch, 1, output_dim]
        """
        # Project features: [Batch, SeqLen, input_dim] → [Batch, SeqLen, hidden_dim]
        x = self.input_projection(x)

        if add_positional_encoding:
            x = self.positional_encoding(x)

        # Self-attention over the time window
        x = self.transformer(x, mask=mask)

        # Mean-pool over time: [Batch, SeqLen, hidden_dim] → [Batch, hidden_dim]
        x = x.mean(dim=1)

        # Project to target: [Batch, output_dim] → [Batch, 1, output_dim]
        # The unsqueeze matches the window_y shape from WindowDataset
        x = self.output_head(x).unsqueeze(1)
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    :param d_model: Model hidden dimension.
    :param max_len: Maximum sequence length to support.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x
