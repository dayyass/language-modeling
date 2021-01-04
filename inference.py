import pickle

import numpy as np

from model import NGramLanguageModel
from utils import get_inference_args

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded


def get_next_token(
    language_model: NGramLanguageModel,
    prefix: str,
    temperature: float = 1.0,
) -> str:
    """
    Sample word using language model, prefix and temperature.
    """
    token2prob = language_model.get_possible_next_tokens(prefix)
    tokens = list(token2prob.keys())
    probs = list(token2prob.values())

    if temperature == 0.0:
        next_token = tokens[probs.index(max(probs))]
    else:
        denominator = sum(prob ** (1 / temperature) for prob in probs)
        probs = [prob ** (1 / temperature) / denominator for prob in probs]
        next_token = np.random.choice(tokens, p=probs)

    return next_token


def generate(
    language_model: NGramLanguageModel,
    prefix: str,
    temperature: float = 1.0,
    max_length: int = 100,
    EOS: str = "<EOS>",
) -> str:
    """
    Generate sentence using language model.
    """
    for _ in range(max_length):
        next_token = get_next_token(language_model, prefix, temperature=temperature)
        prefix += f" {next_token}"
        if (
            prefix.endswith(EOS)
            or len(language_model.get_possible_next_tokens(prefix)) == 0
        ):
            break
    return prefix


if __name__ == "__main__":

    # argparse
    args = get_inference_args()

    # load model
    with open(args.path_to_model, mode="rb") as fp:
        language_model = pickle.load(fp)

    # sequence generation
    generated_sequence = generate(
        language_model=language_model,
        prefix=args.prefix,
        temperature=args.temperature,
        max_length=args.max_length,
        EOS=EOS,
    )

    # print generated sequence
    print(generated_sequence)
