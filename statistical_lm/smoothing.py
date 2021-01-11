from collections import defaultdict
from typing import Counter as CounterType
from typing import DefaultDict, Dict, Set, Tuple

from tqdm import tqdm


def no_smoothing(
    counts: DefaultDict[Tuple[str], CounterType[str]],
) -> DefaultDict[Tuple[str, ...], Dict[str, float]]:
    """
    Transform word counts to probabilities without smoothing.

    :param counts: mapping from (n - 1) previous words to number of times each word occurred after
    :return: mapping from (n - 1) previous words to probability of each word occurred after
    :rtype: DefaultDict[Tuple[str, ...], Dict[str, float]]
    """

    probs: DefaultDict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
    for prefix in tqdm(counts.keys(), desc="transform counts to probabilities"):
        denominator = sum(counts[prefix].values())
        for next_word in counts[prefix].keys():
            probs[prefix][next_word] = counts[prefix][next_word] / denominator
    return probs


def add_k_smoothing(
    counts: DefaultDict[Tuple[str], CounterType[str]],
    delta: float = 1.0,
) -> Tuple[Set[str], DefaultDict[Tuple[str, ...], Dict[str, float]]]:
    """
    Transform word counts to probabilities with add-k (Laplace) smoothing.
    For memory efficiency doesn't save tokens with delta / len(vocab) probability.

    :param counts: mapping from (n - 1) previous words to number of times each word occurred after
    :param float delta: add-k (Laplace) smoothing additive parameter
    :return: vocabulary of all words and
        mapping from (n - 1) previous words to probability of each word occurred after
    :rtype: Tuple[Set[str], DefaultDict[Tuple[str, ...], Dict[str, float]]]
    """

    probs: DefaultDict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
    vocab = set(word for word_counts in counts.values() for word in word_counts)
    for prefix in tqdm(counts.keys(), desc="transform counts to probabilities"):
        denominator = sum(counts[prefix].values()) + delta * len(vocab)
        for next_token in counts[prefix].keys():
            probs[prefix][next_token] = (
                counts[prefix][next_token] + delta
            ) / denominator
    return vocab, probs
