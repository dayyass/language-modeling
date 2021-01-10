from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def str2tensor(
    string: str,
    char2idx: Dict[str, int],
    BOS: str = "<BOS>",
) -> torch.Tensor:
    """
    Transform string to idx tensor using char2idx.

    :param str string: string to transform
    :param char2idx: char to idx mapping
    :param str BOS: begin-of-sentence token (default: "<BOS>")
    :return: idx tensor
    :rtype: torch.Tensor
    """

    string_idx = [char2idx[BOS]] + [char2idx[char] for char in string]
    string_idx_tensor = torch.tensor(string_idx, dtype=torch.long,).unsqueeze(
        0
    )  # batch_size = 1

    return string_idx_tensor


def get_possible_next_tokens(
    model: nn.Module,
    char2idx: Dict[str, int],
    prefix: str,
) -> Dict[str, float]:
    """
    Get tokens distribution after all previous tokens prefix.

    :param nn.Module model: RNN Language Model
    :param Dict[str, int] char2idx: char to idx mapping
    :param str prefix: all previous tokens prefix
    :return: char to probability mapping
    :rtype: Dict[str, float]
    """

    device = next(model.parameters()).device

    prefix_idx_tensor = str2tensor(prefix, char2idx).to(device)
    lengths = torch.tensor([len(prefix)], dtype=torch.long, device=device)

    with torch.no_grad():
        probs = torch.softmax(model(prefix_idx_tensor, lengths), dim=-1)
        probs = probs.squeeze(0)[-1]  # last hidden state
        probs = probs.cpu().numpy()

    char2prob = {char: probs[i] for i, char in enumerate(char2idx.keys())}

    return char2prob


def generate(
    model: nn.Module,
    char2idx: Dict[str, int],
    prefix: str,
    temperature: float = 0.0,
    max_len: int = 100,
    BOS: str = "<BOS>",
    EOS: str = "<EOS>",
) -> str:
    """
    Generate sentence using language model.

    :param RNNLanguageModel model: language model
    :param Dict[str, int] char2idx: char to idx mapping
    :param str prefix: all previous tokens prefix
    :param float temperature: sampling temperature,
        if temperature == 0.0, always takes most likely token - greedy decoding (default: 0.0)
    :param int max_len: max number of generated tokens (chars) (default: 100)
    :param str BOS: begin-of-sentence token (default: "<BOS>")
    :param str EOS: end-of-sentence token (default: "<EOS>")
    :return: generated sequence
    :rtype: str
    """

    for _ in range(max_len):
        char2prob = get_possible_next_tokens(
            model=model, char2idx=char2idx, prefix=prefix
        )
        chars, probs = zip(*char2prob.items())

        if temperature == 0.0:
            next_token = chars[np.argmax(probs)]
        else:
            probs_with_temperature = np.array(
                [prob ** (1.0 / temperature) for prob in probs]
            )
            probs_with_temperature /= sum(probs_with_temperature)
            next_token = np.random.choice(chars, p=probs_with_temperature)

        prefix += next_token

        # BOS to prevent errors
        if (next_token == BOS) or (next_token == EOS) or len(prefix) > max_len:
            break

    return prefix
