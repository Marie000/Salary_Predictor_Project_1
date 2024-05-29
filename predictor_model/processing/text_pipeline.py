import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from predictor_model.config import config
from predictor_model.processing import data_handling

tokenizer = get_tokenizer(config.TOKENIZER)


def _create_vocab(df, column):
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(df[column]), specials=["<unk>", "<pad>"], max_tokens=20000
    )
    return vocab


def create_and_save_vocab(df, column=config.FEATURE):
    vocab = _create_vocab(df, column)
    data_handling.save_vocab(vocab)
    print(f"vocab file saved as {config.SAVE_VOCAB_NAME}")


def text_pipeline(text):
    vocab = data_handling.load_vocab(config.SAVE_VOCAB_NAME)
    return vocab(tokenizer(text))
