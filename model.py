from collections import Counter, defaultdict
from typing import Counter as CounterType
from typing import DefaultDict, Dict, List, Tuple

from nltk import ngrams
from tqdm import tqdm


def count_ngrams(
    data: List[List[str]],
    n: int,
    BOS: str = "<BOS>",
    EOS: str = "<EOS>",
    verbose: bool = True,
) -> DefaultDict[Tuple[str], CounterType[str]]:
    """
    Count how many times each word occurred after (n - 1) previous words.

    :param List[List[str]] data: training data
    :param int n: n-gram order
    :param str BOS: begin-of-sentence token (default: "<BOS>")
    :param str EOS: end-of-sentence token (default: "<EOS>")
    :param bool verbose: verbose (default: True)
    :return: mapping from (n - 1) previous words to number of times each word occurred after
    :rtype: DefaultDict[Tuple[str], CounterType[str]]
    """
    counts: DefaultDict[Tuple[str], CounterType[str]] = defaultdict(Counter)
    if verbose:
        data = tqdm(data, desc="count n-grams")
    for sentence in data:
        sentence = (n - 1) * [BOS] + sentence + [EOS]  # pad sentence beginning with BOS
        for ngram in ngrams(sentence, n):
            prefix = ngram[:-1]
            next_token = ngram[-1]
            counts[prefix].update([next_token])
    return counts


def counts2probs(
    counts: DefaultDict[Tuple[str], CounterType[str]],
) -> DefaultDict[Tuple[str, ...], Dict[str, float]]:
    """
    Transform word counts to probabilities.

    :param counts: mapping from (n - 1) previous words to number of times each word occurred after
    :return: mapping from (n - 1) previous words to probability of each word occurred after
    :rtype: DefaultDict[Tuple[str, ...], Dict[str, float]]
    """
    probs: DefaultDict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
    for prefix in counts.keys():
        denominator = sum(counts[prefix].values())
        for next_word in counts[prefix].keys():
            probs[prefix][next_word] = counts[prefix][next_word] / denominator
    return probs


class NGramLanguageModel:
    """
    N-gram Language Model.
    """

    def __init__(
        self,
        data: List[List[str]],
        n: int,
        BOS: str = "<BOS>",
        EOS: str = "<EOS>",
        verbose: bool = True,
    ):
        """
        Init model with probabilities of word occurred after (n - 1) previous words.

        :param List[List[str]] data: training data
        :param int n: n-gram order
        :param str BOS: begin-of-sentence token (default: "<BOS>")
        :param str EOS: end-of-sentence token (default: "<EOS>")
        :param bool verbose: verbose (default: True)
        """
        self.n = n
        self.BOS = BOS
        self.EOS = EOS

        self.probs = counts2probs(
            count_ngrams(
                data=data,
                n=n,
                BOS=BOS,
                EOS=EOS,
                verbose=verbose,
            )
        )

    def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
        """
        Get words distribution after particular (n - 1) previous words prefix.

        :param str prefix: prefix before sequence generation
        :return: words distribution after particular (n - 1) previous words prefix
        :rtype: Dict[str, float]
        """
        # pad sentence beginning with BOS
        prefix_list = (self.n - 1) * [self.BOS] + prefix.split()
        # fmt: off
        prefix_tuple = tuple(prefix_list[-self.n + 1:])
        # fmt: on
        return self.probs[prefix_tuple]

    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """
        Get probability of particular word occurred after particular (n - 1) previous words.

        :param str prefix: prefix before sequence generation
        :param str next_token: particular word
        :return: probability of particular word occurred after particular (n - 1) previous words
        :rtype: float
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)
