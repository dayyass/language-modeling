import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# TODO: fix
sys.path.append("..")
from utils import load_data  # noqa: E402


def save_data(data: List[List[str]], path: str, verbose: bool = True) -> None:
    """
    Save data.

    :param List[List[str]] data: data to save
    :param str path: path to save
    :param bool verbose: verbose (default: True)
    """
    with open(path, "w") as fp:
        if verbose:
            data = tqdm(data, desc="save data")
        for sentence_list in data:
            sentence = " ".join(sentence_list)
            fp.write(f"{sentence}\n")


def get_args() -> Namespace:
    """
    Argument Parser.

    :return: parsed arguments
    :rtype: Namespace
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="path to data",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        required=True,
        help="test size",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        required=False,
        default=42,
        help="random state",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        required=False,
        default=True,
        help="shuffle",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=True,
        help="verbose",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # argparse
    args = get_args()

    # load data
    data = load_data(path=args.path_to_data, verbose=args.verbose)

    # train_test_split
    train_data, test_data = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=args.shuffle,
    )

    # save data
    filename, ext = os.path.splitext(args.path_to_data)
    save_data(data=train_data, path=f"{filename}_train{ext}")
    save_data(data=test_data, path=f"{filename}_test{ext}")
