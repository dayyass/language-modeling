import json
import os
from typing import Dict

import numpy as np
import torch

from arg_parse import get_inference_args  # isort:skip
from model import RNNLanguageModel  # isort:skip
from utils import set_global_seed, str2tensor  # isort:skip

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded


def get_possible_next_tokens(
    model: RNNLanguageModel,
    char2idx: Dict[str, int],
    prefix: str,
) -> Dict[str, float]:
    """
    Get tokens distribution after all previous tokens prefix.

    :param RNNLanguageModel model: RNN Language Model
    :param Dict[str, int] char2idx: char to idx mapping
    :param str prefix: all previous tokens prefix
    :return: char to probability mapping
    :rtype: Dict[str, float]
    """

    model.eval()

    device = next(model.parameters()).device

    prefix_idx_tensor = str2tensor(prefix, char2idx).to(device)
    lengths = torch.tensor(
        [len(prefix) + 1], dtype=torch.long, device=device
    )  # add BOS

    with torch.no_grad():
        probs = torch.softmax(model(prefix_idx_tensor, lengths), dim=-1)
        probs = probs.squeeze(0)[-1]  # last hidden state
        probs = probs.cpu().numpy()

    char2prob = {char: probs[i] for i, char in enumerate(char2idx.keys())}

    return char2prob


def get_next_token_prob(
    model: RNNLanguageModel,
    char2idx: Dict[str, int],
    prefix: str,
    next_token: str,
) -> float:
    """
    Get probability of particular token occurred after all previous tokens.

    :param RNNLanguageModel model: RNN Language Model
    :param Dict[str, int] char2idx: char to idx mapping
    :param str prefix: all previous tokens prefix
    :param str next_token: particular token
    :return: probability of particular token occurred after all previous tokens
    :rtype: float
    """

    model.eval()

    char2prob = get_possible_next_tokens(
        model=model,
        char2idx=char2idx,
        prefix=prefix,
    )

    next_token_prob = char2prob.get(next_token, 0.0)

    return next_token_prob


def get_next_token(
    model: RNNLanguageModel,
    char2idx: Dict[str, int],
    prefix: str,
    temperature: float = 0.0,
) -> str:
    """
    Sample word using language model, prefix and temperature.

    :param RNNLanguageModel model: language model
    :param Dict[str, int] char2idx: char to idx mapping
    :param str prefix: prefix before sequence generation
    :param float temperature: sampling temperature,
        if temperature == 0.0, always takes most likely token - greedy decoding (default: 0.0)
    :return: next token
    :rtype: str
    """

    model.eval()

    char2prob = get_possible_next_tokens(model=model, char2idx=char2idx, prefix=prefix)
    chars, probs = zip(*char2prob.items())

    if temperature == 0.0:
        next_token = chars[np.argmax(probs)]
    else:
        probs_with_temperature = np.array(
            [prob ** (1.0 / temperature) for prob in probs]
        )
        probs_with_temperature /= sum(probs_with_temperature)
        next_token = np.random.choice(chars, p=probs_with_temperature)

    return next_token


def generate(
    model: RNNLanguageModel,
    char2idx: Dict[str, int],
    prefix: str,
    temperature: float = 0.0,
    max_length: int = 100,
) -> str:
    """
    Generate sentence using language model.

    :param RNNLanguageModel model: language model
    :param Dict[str, int] char2idx: char to idx mapping
    :param str prefix: all previous tokens prefix
    :param float temperature: sampling temperature,
        if temperature == 0.0, always takes most likely token - greedy decoding (default: 0.0)
    :param int max_length: max number of generated tokens (chars) (default: 100)
    :return: generated sequence
    :rtype: str
    """

    model.eval()

    for _ in range(max_length):
        next_token = get_next_token(
            model=model,
            char2idx=char2idx,
            prefix=prefix,
            temperature=temperature,
        )
        prefix += next_token

        # BOS to prevent errors
        if (next_token == BOS) or (next_token == EOS):
            break

    return prefix


if __name__ == "__main__":

    # argparse
    args = get_inference_args()

    # set seed and device
    set_global_seed(args.seed)
    device = torch.device(args.device)

    # load

    # # vocab char2idx
    path = os.path.join(args.path_to_model_folder, "vocab.json")
    with open(path, mode="r") as fp:
        char2idx = json.load(fp)

    # # model parameters
    path = os.path.join(args.path_to_model_folder, "model_parameters.json")
    with open(path, mode="r") as fp:
        model_parameters = json.load(fp)

    # # model state_dict
    path = os.path.join(args.path_to_model_folder, "language_model.pth")
    state_dict = torch.load(path, map_location=device)

    # init model
    model = RNNLanguageModel(**model_parameters)
    model.load_state_dict(state_dict)
    model.eval()

    # sequence generation
    generated_sequence = generate(
        model=model,
        char2idx=char2idx,
        prefix=args.prefix,
        temperature=args.temperature,
        max_length=args.max_length,
    )

    # print generated sequence
    print(generated_sequence)
