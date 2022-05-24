from collections import Counter, defaultdict
from typing import Counter as CounterType
from typing import DefaultDict, Dict, List, Tuple

from nltk import ngrams
from tqdm import tqdm

from smoothing import add_k_smoothing, no_smoothing  # isort:skip

BOS = "<BOS>"
EOS = "<EOS>"


def count_ngrams(
    data: List[List[str]],
    n: int,
    verbose: bool = True,
) -> DefaultDict[Tuple[str], CounterType[str]]:
    """
    Count how many times each word occurred after (n - 1) previous words.

    :param List[List[str]] data: training data
    :param int n: n-gram order
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


class NGramLanguageModel:
    """
    N-gram Language Model.
    """

    def __init__(
        self,
        data: List[List[str]],
        n: int,
        verbose: bool = True,
    ):
        """
        Init model with probabilities of word occurred after (n - 1) previous words.

        :param List[List[str]] data: training data
        :param int n: n-gram order
        :param bool verbose: verbose (default: True)
        """

        self.n = n
        self.BOS = BOS
        self.EOS = EOS

        self.probs = no_smoothing(
            count_ngrams(
                data=data,
                n=n,
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

        if self.n == 1:  # unigram model case
            prefix_tuple = ()
        else:
            # pad sentence beginning with BOS
            prefix_list = (self.n - 1) * [self.BOS] + prefix.split()
            # fmt: off
            prefix_tuple = tuple(prefix_list[-self.n + 1:])  # type: ignore
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

        return self.get_possible_next_tokens(prefix).get(next_token, 0.0)


# TODO: validate it with correct inheritence
class LaplaceLanguageModel(NGramLanguageModel):
    """
    N-gram Language Model with add-k (Laplace) smoothing.
    """

    # no super().__init__()
    def __init__(
        self,
        data: List[List[str]],
        n: int,
        delta: float = 1.0,
        verbose: bool = True,
    ):
        """
        Init model with probabilities of word occurred after (n - 1) previous words.

        :param List[List[str]] data: training data
        :param int n: n-gram order
        :param float delta: add-k (Laplace) smoothing additive parameter (default: 1.0)
        :param bool verbose: verbose (default: True)
        """

        self.n = n
        self.delta = delta
        self.BOS = BOS
        self.EOS = EOS

        self.vocab, self.probs = add_k_smoothing(
            count_ngrams(
                data=data,
                n=n,
                verbose=verbose,
            ),
            delta=delta,
        )

    def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
        """
        Get words distribution after particular (n - 1) previous words prefix.

        :param str prefix: prefix before sequence generation
        :return: words distribution after particular (n - 1) previous words prefix
        :rtype: Dict[str, float]
        """

        probs = super().get_possible_next_tokens(prefix)

        missing_prob_total = 1.0 - sum(probs.values())
        missing_prob_total = max(0.0, missing_prob_total)  # prevent rounding errors

        missing_prob = missing_prob_total / max(1, len(self.vocab) - len(probs))

        add_k_probs = {token: probs.get(token, missing_prob) for token in self.vocab}

        return add_k_probs

    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """
        Get probability of particular word occurred after particular (n - 1) previous words.

        :param str prefix: prefix before sequence generation
        :param str next_token: particular word
        :return: probability of particular word occurred after particular (n - 1) previous words
        :rtype: float
        """

        probs = super().get_possible_next_tokens(prefix)

        if next_token in probs:
            next_token_prob = probs[next_token]
        else:
            missing_prob_total = 1.0 - sum(probs.values())
            missing_prob_total = max(0.0, missing_prob_total)  # prevent rounding errors
            missing_prob = missing_prob_total / max(1, len(self.vocab) - len(probs))
            next_token_prob = missing_prob

        return next_token_prob
