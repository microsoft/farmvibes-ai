from typing import Any, List, Tuple, Union

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from .encoder import Encoder
from .locally_connected import LocallyConnected1d


class MyLSTM(nn.LSTM):
    def forward(self, *args: Any, **kwargs: Any):
        return super().forward(*args, **kwargs)[0]


class DeepMCModel(nn.Module):
    def __init__(
        self,
        first_channels: int,  # 3
        rest_channels: int,  # 1
        first_encoder_channels: int,  # 3
        rest_encoder_channels: Tuple[int, int, int],  # [4, 8, 16]
        sequence_length: int,  # 24
        kernel_size: int,  # 2
        num_inputs: int,  # 6
        encoder_layers: int = 2,
        encoder_features: int = 4,
        encoder_heads: int = 4,
        encoder_ff_features: int = 16,
        encoder_dropout: float = 0.1,
        decoder_features: Tuple[int, int] = (20, 16),
        dropout: float = 0.2,
        batch_first: bool = True,
        return_sequence: bool = True,
    ):
        super(DeepMCModel, self).__init__()
        self.return_sequence = return_sequence
        self.num_inputs = num_inputs
        out_seq_len = sequence_length - kernel_size + 1
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange("b l d -> b d l"),
                    LocallyConnected1d(
                        in_channels=first_channels,
                        out_channels=first_encoder_channels,
                        seq_len=sequence_length,
                        kernel_size=kernel_size,
                    ),
                    nn.BatchNorm1d(first_encoder_channels),
                    Rearrange("b d l -> b l d"),
                    Encoder(
                        in_features=first_encoder_channels,
                        num_layers=encoder_layers,
                        d_model=encoder_features,
                        num_heads=encoder_heads,
                        d_ff=encoder_ff_features,
                        max_seq_len=out_seq_len,
                        dropout=encoder_dropout,
                    ),
                    nn.Flatten(),
                )
            ]
        )

        re1, re2, re3 = rest_encoder_channels
        for _ in range(num_inputs - 1):
            self.encoders.append(
                nn.Sequential(
                    Rearrange("b l d -> b d l"),
                    LocallyConnected1d(
                        in_channels=rest_channels,
                        out_channels=re1,
                        seq_len=sequence_length,
                        kernel_size=kernel_size,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(re1),
                    LocallyConnected1d(
                        in_channels=re1,
                        out_channels=re2,
                        seq_len=out_seq_len,
                        kernel_size=kernel_size,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(re2),
                    Rearrange("b d l -> b l d"),
                    MyLSTM(
                        input_size=re2,
                        hidden_size=re3,
                        num_layers=1,
                        batch_first=batch_first,
                        dropout=dropout,
                    ),
                    # nn.ReLU(),  # Do ReLU outside the model
                )
            )

        dec_input_features = out_seq_len * encoder_features + (self.num_inputs - 1) * re3
        df1, df2 = decoder_features
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(dec_input_features),
            Rearrange("b d -> b 1 d"),
            MyLSTM(
                input_size=dec_input_features,
                hidden_size=df1,
                batch_first=batch_first,
                dropout=dropout,
            ),
            Rearrange("b 1 d -> b d"),
            nn.ReLU(),
            nn.BatchNorm1d(df1),
            nn.Linear(df1, df2),
            nn.ReLU(),
            nn.Linear(df2, 1),
        )

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        sliced_encoders = nn.ModuleList(list(self.encoders)[1:])
        x = [self.encoders[0](x[0])] + [
            F.relu(encoder(xi)[:, -1]) for encoder, xi in zip(sliced_encoders, x[1:])
        ]
        x = torch.cat(x, dim=1)
        x = self.decoder(x)
        return x
