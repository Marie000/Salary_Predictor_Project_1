from predictor_model.config import config
from predictor_model.processing import data_handling, text_pipeline
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


vocab = data_handling.load_vocab()
BATCH_SIZE = config.BATCH_SIZE


def _collate_with_padding(data):
    text_list, result_list = [], []
    for text, result in data:
        result_list.append(result)
        text = torch.tensor(text_pipeline.text_pipeline(text), dtype=torch.int64)
        text_list.append(text)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=padding_index)
    result_list = torch.tensor(result_list, dtype=torch.float32, device=device)

    return text_list, result_list


def create_dataloader(data):
    dataloader = Dataloader(
        data, collate_fn=_collate_with_padding, batch_size=BATCH_SIZE, drop_last=True
    )
    return dataloader
