import pickle
from typing import List

import numpy as np
from nltk import ngrams
from tqdm import tqdm

from arg_parse import get_validate_args  # isort:skip
from model import NGramLanguageModel  # isort:skip
from utils import load_data  # isort:skip

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded


def compute_perplexity(
    language_model: NGramLanguageModel,
    data: List[List[str]],
    min_logprob: float = np.log(10 ** -50.0),
) -> float:
    """
    Compute perplexity using language model and validation data.

    :param NGramLanguageModel language_model: language model
    :param List[List[str]] data: validation data
    :param float min_logprob: if logprob is smaller than min_logprob,
        set it equal to min_logrob (default: np.log(10 ** -50.0))
    :return: perplexity
    :rtype: float
    """

    log_likelihood, N = 0, 0

    for sentence in tqdm(data, desc="compute perplexity"):
        N += len(sentence) + 1
        sentence = (
            (language_model.n - 1) * [BOS] + sentence + [EOS]
        )  # pad sentence beginning with BOS

        for ngram in ngrams(sentence, language_model.n):
            prefix = ngram[:-1]
            next_token = ngram[-1]

            prob = language_model.get_next_token_prob(" ".join(prefix), next_token)

            log_prob = max(min_logprob, np.log(prob) if prob != 0.0 else min_logprob)
            log_likelihood += log_prob

    perplexity = np.exp(-1 / N * log_likelihood)

    return perplexity


if __name__ == "__main__":

    # argparse
    args = get_validate_args()

    # load data
    data = load_data(path=args.path_to_data, verbose=args.verbose)

    # load model
    with open(args.path_to_model, mode="rb") as fp:
        language_model = pickle.load(fp)

    # validate
    perplexity = compute_perplexity(
        language_model=language_model,
        data=data,
    )

    # print perplexity
    print(f"perplexity of {language_model.n}-gram language model: {perplexity}")
