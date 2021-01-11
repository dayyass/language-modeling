import json
import os
from typing import Dict, List

import numpy as np
import torch
from inference import get_next_token_prob
from model import RNNLanguageModel
from tqdm import tqdm
from utils import get_validate_args, load_data, set_global_seed

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded


def compute_perplexity(
    language_model: RNNLanguageModel,
    char2idx: Dict[str, int],
    data: List[str],
    min_logprob: float = np.log(10 ** -50.0),
    EOS: str = "<EOS>",
) -> float:
    """
    Compute perplexity using RNN language model and validation data.

    :param RNNLanguageModel language_model: language model
    :param Dict[str, int] char2idx: char to idx mapping
    :param List[str] data: validation data
    :param float min_logprob: if logprob is smaller than min_logprob,
        set it equal to min_logrob (default: np.log(10 ** -50.0))
    :param str EOS: end-of-sentence token (default: "<EOS>")
    :return: perplexity
    :rtype: float
    """

    log_likelihood, N = 0, 0
    for sentence in tqdm(data, desc="compute perplexity"):
        N += len(sentence) + 1
        for i in range(len(sentence) + 1):  # for EOS
            if i != len(sentence):
                prefix = sentence[:i]
                next_token = sentence[i]
            else:
                prefix = sentence
                next_token = EOS
            prob = get_next_token_prob(
                model=language_model,
                char2idx=char2idx,
                prefix=prefix,
                next_token=next_token,
            )
            log_prob = max(min_logprob, np.log(prob) if prob != 0.0 else min_logprob)
            log_likelihood += log_prob
    perplexity = np.exp(-1 / N * log_likelihood)
    return perplexity


if __name__ == "__main__":

    # argparse
    args = get_validate_args()

    # set seed and device
    set_global_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = load_data(path=args.path_to_data, verbose=args.verbose)

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

    # validate
    perplexity = compute_perplexity(
        language_model=model,
        char2idx=char2idx,
        data=data,
        EOS=EOS,
    )

    # print perplexity
    print(f"perplexity of RNN language model: {perplexity}")
