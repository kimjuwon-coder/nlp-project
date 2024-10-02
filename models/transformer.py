import numpy as np
import torch
from torch import nn

# incomplete

def positional_encoding(x):
    height = x.size(1)
    d_model = x.size(2)

    angles = torch.arange(d_model)
    angles[1::2] -= 1
    angles = angles.view(1, -1) / d_model
    angles = torch.arange(height).view(-1, 1) / torch.pow(10000, angles)

    output = torch.stack([angles[:, ::2].sin(), angles[:, 1::2].cos()], dim=2)
    output = output.view(height, d_model)
    output = x + output

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        self.layer_query = nn.Linear(input_dim, input_dim)
        self.layer_key = nn.Linear(input_dim, input_dim)
        self.layer_value = nn.Linear(input_dim, input_dim)
        self.layer_output = nn.Linear(input_dim, input_dim)
        return

    def forward(self, query, key, value, mask=None):
        output_size = query.size()
        query = self.layer_query(query)  # (-1 x T x d)
        key = self.layer_key(key)  # (-1 x T x d)
        value = self.layer_value(value)  # (-1 x T x d)

        query = query.view(*output_size[:2], self.num_heads, self.d_k)  # (-1 x T x num_heads x d_k)
        query = query.permute(0, 2, 1, 3)  # (-1 x num_heads x T x d_k)
        key = key.view(*output_size[:2], self.num_heads, self.d_k)  # (-1 x T x num_heads x d_k)
        key = key.permute(0, 2, 1, 3)  # (-1 x num_heads x T x d_k)
        value = value.view(*output_size[:2], self.num_heads, self.d_k)  # (-1 x T x num_heads x d_k)
        value = value.permute(0, 2, 1, 3)  # (-1  xnum_heads x T x d_k)

        self_att_score = torch.einsum("ijkm,ijlm->ijkl", query, key) / np.sqrt(self.d_k)  # (-1 x num_heads x T x T)
        if mask is not None:
            self_att_score += mask * -1e9
        self_att_score = self_att_score.softmax(dim=-1)  # (-1 x num_heads x T x T)

        output = torch.einsum(
            "ijkm,ijml->ijkl", self_att_score, value
        )  # (-1 x num_heads x T x d_k)
        output = output.permute(0, 2, 1, 3).contiguous()  # (-1 x T x num_heads x d_k)
        output = output.view(*output_size)  # (-1 x T x d)
        output = self.layer_output(output)

        return self_att_score, output


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

        return

    def forward(self, x, padding_mask):
        _, output = self.attention(x, x, x, padding_mask)
        output = x + self.dropout(output)
        output = self.layer_norm(output)

        output = output + self.layer_linear(output)
        output = self.layer_norm(output)

        return output


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
                EncoderLayer(input_dim, hidden_dim, num_heads, dropout)
                    for _ in range(num_layers)
            ])

        return

    def forward(self, x, padding_mask=None):
        output = positional_encoding(x)
        output = self.dropout(output)

        for layer in self.encoder_layers:
            output = layer(output, padding_mask)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.masked_attention = MultiHeadAttention(input_dim, num_heads)
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        self.layer_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

        return

    def forward(self, x, output_enc, target_mask, padding_mask):
        _, query = self.masked_attention(x, x, x, mask=target_mask)
        query = self.layer_norm_1(x + query)

        output = self.attention(query, output_enc, output_enc, mask=padding_mask)
        output = query + self.dropout(output)
        output = self.layer_norm_2(output)

        output = output + self.layer_linear(output)
        output = self.layer_norm_3(output)

        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([
                DecoderLayer(input_dim, hidden_dim, num_heads, dropout)
                    for _ in range(num_layers)
            ])

        return

    def forward(self, x, output_enc, target_mask, padding_mask):
        output = positional_encoding(x)
        output = self.dropout(output)

        for layer in self.decoder_layers:
            output = layer(output, output_enc, target_mask, padding_mask)

        return output


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = Decoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        return

    def forward(self, x_enc, x_dec):
        padding_mask = self.generate_padding_mask(x_enc)
        target_mask = self.generate_padding_mask(x_dec)

        output_enc = self.encoder(x_enc, padding_mask)
        output = self.decoder(x_dec, output_enc, target_mask, padding_mask)

        return output

    def generate_padding_mask(self, x):

        return

    def generate_target_mask(self, x):

        return