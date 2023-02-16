from torch import nn

from notebook_lib.helpers import point_wise_feed_forward_network, positional_encoding
from notebook_lib.transform import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(
            in_features=d_model, out_features=d_model, d_ff=d_ff
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Sequential(nn.Linear(in_features, self.d_model), nn.ReLU())
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = x * self.d_model**0.5
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask)

        return x  # (batch_size, input_seq_len, d_model)
