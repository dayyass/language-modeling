from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


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
        :param Optional[int] max_len: max sentence length
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
