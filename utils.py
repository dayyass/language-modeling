from argparse import ArgumentParser, Namespace
from typing import List

from tqdm import tqdm


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path to train data")
    parser.add_argument("--n", type=int, required=True, help="n-gram order")
    parser.add_argument(
        "--path_to_save",
        type=str,
        required=False,
        default="models/language_model.pkl",
        help="path to save model",
    )
    parser.add_argument(
        "--verbose", type=bool, required=False, default=True, help="verbose"
    )
    args = parser.parse_args()
    return args


def load_data(path: str, verbose: bool = True) -> List[List[str]]:
    """
    Load data.
    """
    data = []
    with open(path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp, desc="load data")
        for line in fp:
            data.append(line.split())
    return data
