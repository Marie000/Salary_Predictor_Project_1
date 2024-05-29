import torch
from torch import nn
from predictor_model.processing import data_handling
from predictor_model.config import config

## NOTE TO SELF: most of this goes in the training_pipeline file. 


vocab_size = data_handling.get_vocab_size()


class RNNModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, dropout=0.5
        )
        self.fc = nn.Linear(hidden_dim * num_layers, 1)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.fc.weight.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        x = x.permute(1, 0)
        emb = self.embedding(x)
        # output will not be used because we have a many-to-one rnn
        output, (hidden, cell) = self.rnn(emb)
        hidden.squeeze_(0)
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(-1, self.hidden_dim * self.num_layers)
        out = self.fc(hidden)
        return out


def create_model():
    model = RNNModel(
        vocab_size,
        config.EMBEDDING_DIM,
        config.HIDDEN_DIM,
        config.NUM_LAYERS,
        config.DROPOUT_RATE,
    )
    return model
