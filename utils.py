import random
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_args() -> Namespace:
    """
    Training Argument Parser.

    :return: parsed arguments
    :rtype: Namespace
    """

    parser = ArgumentParser()

    # path
    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="path to train data",
    )
    parser.add_argument(
        "--path_to_save_folder",
        type=str,
        required=False,
        default="models/rnn_language_model",
        help="path to save folder",
    )

    # dataset and dataloader
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=None,
        help="max sentence length (chars)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="dataloader batch_size",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        required=False,
        default=True,
        help="dataloader shuffle",
    )

    # model
    parser.add_argument(
        "--embedding_dim",
        type=int,
        required=True,
        help="embedding dimension",
    )
    parser.add_argument(
        "--rnn_hidden_size",
        type=int,
        required=True,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--rnn_num_layers",
        type=int,
        required=False,
        default=1,
        help="number of LSTM layers",
    )
    parser.add_argument(
        "--rnn_dropout",
        type=float,
        required=False,
        default=0.0,
        help="LSTM dropout",
    )

    # train
    parser.add_argument(
        "--train_eval_freq",
        type=int,
        required=False,
        default=50,
        help="evaluation frequency (number of batches)",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        required=False,
        default=1.0,
        help="max_norm parameter in clip_grad_norm",
    )

    # additional
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        choices=["cpu", "cuda"],
        help="torch device (available: 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=True,
        help="verbose",
    )

    args = parser.parse_args()
    return args


def load_data(path: str, verbose: bool = True) -> List[str]:
    """
    Load data.

    :param str path: path to load data
    :param bool verbose: verbose (default: True)
    :return: data
    :rtype: List[str]
    """

    data = []
    with open(path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp, desc="load data")
        for line in fp:
            data.append(line)
    return data


def get_char2idx(
    data: List[str],
    BOS: str = "<BOS>",
    EOS: str = "<EOS>",
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Get char to idx mapping for PyTorch models.
    :param List[str] data: data
    :param str BOS: begin-of-sentence token (default: "<BOS>")
    :param str EOS: end-of-sentence token (default: "<EOS>")
    :param bool verbose: verbose (default: True)
    :return: char to idx mapping
    :rtype: Dict[str, int]
    """

    char2idx: Dict[str, int] = {}

    char2idx[BOS] = len(char2idx)
    char2idx[EOS] = len(char2idx)

    if verbose:
        data = tqdm(data, desc="prepare char2idx")
    for sentence in data:
        for char in sentence:
            if char not in char2idx:
                char2idx[char] = len(char2idx)
    return char2idx


class LMDataset(Dataset):
    """
    Dataset for Language Modeling.
    """

    def __init__(
        self,
        data: List[str],
        char2idx: Dict[str, int],
        max_len: Optional[int] = None,
        BOS: str = "<BOS>",
        EOS: str = "<EOS>",
        verbose: bool = True,
    ):
        """
        Init dataset.

        :param List[str] data: data
        :param Dict[str, int] char2idx: char to idx mapping
        :param Optional[int] max_len: max sentence length (chars)
        :param str BOS: begin-of-sentence token (default: "<BOS>")
        :param str EOS: end-of-sentence token (default: "<EOS>")
        :param bool verbose: verbose (default: True)
        """

        self.char2idx = char2idx
        self.data = []
        if verbose:
            data = tqdm(data, desc="prepare dataset")
        for sentence in data:
            sentence_idx = [char2idx[char] for char in sentence]
            if max_len is not None:
                sentence_idx = sentence_idx[:max_len]
            self.data.append(
                [char2idx[BOS]] + sentence_idx + [char2idx[EOS]],
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)


class LMCollator(object):
    """
    Collator that handles variable-size sentences.
    """

    def __init__(self, padding_value: int):
        """
        Init collator.

        :param int padding_value: padding value
        """

        self.padding_value = padding_value

    def __call__(
        self,
        sentences: List[torch.Tensor],
    ) -> torch.Tensor:

        sentences = pad_sequence(
            sentences,
            batch_first=True,
            padding_value=self.padding_value,
        )

        return sentences


def infer_lengths(
    sentences: torch.Tensor,
    bos_id: int,
    eos_id: int,
) -> torch.Tensor:
    """
    Compute length of each sentence in sentences (excl. EOS).

    :param torch.Tensor sentences: sentences of shape [batch_size, seq_len]
    :param int bos_id: begin-of-sentence token index
    :param int eos_id: end-of-sentence token index
    :return: length of each sentence in sentences (excl. EOS) of shape [batch_size]
    :rtype: torch.Tensor
    """

    lengths = torch.sum(
        torch.logical_and(sentences != bos_id, sentences != eos_id),
        dim=-1,
    )
    return lengths


def masking(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths tensor to binary mask to compute loss without pad tokens.
    implement: https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch

    :param torch.Tensor lengths: lengths of each sentence
    :return: binary mask to compute loss without pad tokens
    :rtype: torch.Tensor
    """

    device = lengths.device
    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    return torch.arange(end=max_len, device=device).expand(
        size=(lengths_shape, max_len)
    ) < lengths.unsqueeze(1)
