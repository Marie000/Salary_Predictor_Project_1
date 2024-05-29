import os
import pandas as pd
from predictor_model.config import config
import torch


def load_dataset(filename):
    filepath = os.path.join(config.DATA_PATH, filename)
    _data = pd.read_csv(filepath)
    return _data


def save_model(model):
    torch.save(
        model,
        os.path.join(config.SAVE_MODEL_PATH, config.SAVE_MODEL_NAME),
    )
    print(f"model saved as ${config.SAVE_MODEL_NAME}")
    return


def save_vocab(vocab, filename=config.SAVE_MODEL_NAME):
    torch.save(
        vocab,
        os.path.join(config.SAVE_MODEL_PATH, config.SAVE_VOCAB_NAME),
    )
    print(f"vocab saved as ${config.SAVE_VOCAB_NAME}")
    return


def load_model(name=config.SAVE_MODEL_NAME):
    _model = torch.load(os.path.join(config.SAVE_MODEL_PATH, name))
    return _model


def load_vocab(name=config.SAVE_VOCAB_NAME):
    _vocab = torch.load(os.path.join(config.SAVE_MODEL_PATH, name))
    return _vocab


def get_vocab_size(name=config.SAVE_VOCAB_NAME):
    _vocab = torch.load(os.path.join(config.SAVE_MODEL_PATH, name))
    return len(_vocab)
