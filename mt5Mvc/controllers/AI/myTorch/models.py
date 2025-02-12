import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        d_model:      dimension of embeddings
        dropout:      randomly zeroes-out some of the input
        max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()
        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)
        # create position column
        position = torch.arange(0, max_length).unsqueeze(1)
        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # calc sine on even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # add dimension
        pe = pe.unsqueeze(0)
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: embeddings (batch_size, seq_length, d_model)
        Returns: embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # perform dropout
        return self.dropout(x)


class InputEmbeddings(nn.Module):
    """
    https://medium.com/@bavalpreetsinghh/transformer-from-scratch-using-pytorch-28a5d1b2e033
    """

    def __init__(self, d_model, data_size: int):
        super().__init__()
        self.d_model = d_model
        self.data_size = data_size
        self.embedding = nn.Embedding(data_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class LayerNormalization(nn.Module):
    """
    https://medium.com/@bavalpreetsinghh/transformer-from-scratch-using-pytorch-28a5d1b2e033
    """

    def __init__(self, features: int, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        :param x: (batch, seq_len, hidden_size)
        """
        # keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=16, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.Linear
