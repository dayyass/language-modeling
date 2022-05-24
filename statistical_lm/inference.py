import pickle

import numpy as np
from scipy.special import softmax

from arg_parse import get_inference_args  # isort:skip
from model import NGramLanguageModel  # isort:skip
from utils import set_global_seed  # isort:skip

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded


def get_next_token(
    language_model: NGramLanguageModel,
    prefix: str,
    strategy: str,
    temperature: float = 0.0,
    k: int = 10,
) -> str:
    """
    Sample word using language model, prefix and temperature.

    :param NGramLanguageModel language_model: language model
    :param str prefix: prefix before sequence generation
    :param str strategy: decoding strategy
        (available: 'sampling', 'top-k-uniform', 'top-k', 'top-p-uniform', 'top-p' and 'beam search')
        (default: 'sampling')
    :param float temperature: sampling temperature,
        if temperature == 0.0, always takes most likely token - greedy decoding
        (only for 'sampling' decoding strategy) (default: 0.0)
    :param int k: top-k parameter (only for 'top-k-uniform' and 'top-k' decoding strategy) (default: 10)
    :return: next token
    :rtype: str
    """

    token2prob = language_model.get_possible_next_tokens(prefix)
    tokens, probs = zip(*token2prob.items())

    if strategy == "sampling":

        if temperature == 0.0:
            next_token = tokens[np.argmax(probs)]
        else:
            probs_with_temperature = np.array(
                [prob ** (1.0 / temperature) for prob in probs]
            )
            probs_with_temperature /= sum(probs_with_temperature)
            next_token = np.random.choice(tokens, p=probs_with_temperature)

    elif strategy == "top-k-uniform":

        topk_idx = np.argsort(probs)[-k:]
        topk_tokens = np.array(tokens)[topk_idx].tolist()

        next_token = np.random.choice(topk_tokens)

    elif strategy == "top-k":

        zero_prob_idx = np.argsort(probs)[:-k]

        top_k_probs = np.array(probs)
        top_k_probs[zero_prob_idx] = 0

        with np.errstate(divide="ignore"):  # ignore np.log(0) warning
            top_k_probs = softmax(np.log(top_k_probs))

        next_token = np.random.choice(tokens, p=top_k_probs)

    elif strategy == "top-p-uniform":  # TODO
        raise NotImplementedError()

    elif strategy == "top-p":  # TODO
        raise NotImplementedError()

    elif strategy == "beam search":  # TODO
        raise NotImplementedError()

    else:
        raise ValueError(
            f"{strategy} strategy is not known, "
            "use 'sampling', 'top-k-uniform', 'top-k', 'top-p-uniform', 'top-p' or 'beam search'"
        )

    return next_token


def generate(
    language_model: NGramLanguageModel,
    prefix: str,
    strategy: str,
    temperature: float = 0.0,
    k: int = 10,
    max_length: int = 100,
) -> str:
    """
    Generate sentence using language model.

    :param NGramLanguageModel language_model: language model
    :param str prefix: prefix before sequence generation
    :param str strategy: decoding strategy
        (available: 'sampling', 'top-k-uniform', 'top-k', 'top-p-uniform', 'top-p' and 'beam search')
        (default: 'sampling')
    :param float temperature: sampling temperature,
        if temperature == 0.0, always takes most likely token - greedy decoding
        (only for 'sampling' decoding strategy) (default: 0.0)
    :param int k: top-k parameter (only for 'top-k-uniform' and 'top-k' decoding strategy) (default: 10)
    :param int max_length: max number of generated words (default: 100)
    :return: generated sequence
    :rtype: str
    """

    for _ in range(max_length):
        next_token = get_next_token(
            language_model,
            prefix,
            strategy=strategy,
            temperature=temperature,
            k=k,
        )
        prefix += f" {next_token}"

        if (next_token == EOS) or (
            len(language_model.get_possible_next_tokens(prefix)) == 0
        ):
            break

    return prefix


if __name__ == "__main__":

    # argparse
    args = get_inference_args()

    # seed
    set_global_seed(args.seed)

    # load model
    with open(args.path_to_model, mode="rb") as fp:
        language_model = pickle.load(fp)

    # sequence generation
    generated_sequence = generate(
        language_model=language_model,
        prefix=args.prefix,
        strategy=args.strategy,
        temperature=args.temperature,
        k=args.k,
        max_length=args.max_length,
    )

    # print generated sequence
    print(generated_sequence)
