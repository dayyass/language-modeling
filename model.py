from collections import Counter, defaultdict
from typing import Counter as CounterType
from typing import DefaultDict, Dict, List, Tuple

from nltk import ngrams
from tqdm import tqdm


def count_ngrams(
    data: List[List[str]],
    n: int,
    BOS: str,
    EOS: str,
    verbose: bool = True,
) -> DefaultDict[Tuple[str], CounterType[str]]:
    """
    Count how many times each word occurred after (n - 1) previous words.
    """
    counts: DefaultDict[Tuple[str], CounterType[str]] = defaultdict(Counter)
    if verbose:
        data = tqdm(data, desc="count n-grams")
    for sentence in data:
        sentence = (n - 1) * [BOS] + sentence + [EOS]  # pad sentence beginning with BOS
        for ngram in ngrams(sentence, n):
            counts[ngram[:-1]].update([ngram[-1]])
    return counts


def counts2probs(
    counts: DefaultDict[Tuple[str], CounterType[str]],
) -> DefaultDict[Tuple[str, ...], Dict[str, float]]:
    """
    Transform word counts to probabilities.
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
        BOS: str,
        EOS: str,
        verbose: bool = True,
    ):
        """
        Init model with probabilities of word occurred after (n - 1) previous words.
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
        """
        # pad sentence beginning with BOS
        prefix_list = (self.n - 1) * [self.BOS] + prefix.split()
        # fmt: off
        prefix_tuple = tuple(prefix_list[-self.n:])
        # fmt: on
        return self.probs[prefix_tuple]

    def get_next_token_prob(self, prefix: str, next_token: str):
        """
        Get probability of particular word occurred after particular (n - 1) previous words.
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)
