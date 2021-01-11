from argparse import ArgumentParser, Namespace
from typing import List

from tqdm import tqdm


def load_data(path: str, verbose: bool = True) -> List[List[str]]:
    """
    Load data.

    :param str path: path to load data
    :param bool verbose: verbose (default: True)
    :return: data
    :rtype: List[List[str]]
    """

    data = []
    with open(path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp, desc="load data")
        for line in fp:
            data.append(line.split())
    return data


def get_train_args() -> Namespace:
    """
    Training Argument Parser.

    :return: parsed arguments
    :rtype: Namespace
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="path to train data",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="n-gram order",
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        required=False,
        default="models/language_model.pkl",
        help="path to save model",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        required=False,
        default=None,
        choices=[None, "add-k"],
        help="smoothing method (available: None, 'add-k')",
    )
    parser.add_argument(
        "--delta",
        type=float,
        required=False,
        default=1.0,
        help="add-k smoothing additive parameter",
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


def get_validate_args() -> Namespace:
    """
    Validation Argument Parser.

    :return: parsed arguments
    :rtype: Namespace
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="path to train data",
    )
    parser.add_argument(
        "--path_to_model",
        type=str,
        required=True,
        help="path to language model",
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


def get_inference_args() -> Namespace:
    """
    Inference Argument Parser.

    :return: parsed arguments
    :rtype: Namespace
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_model",
        type=str,
        required=True,
        help="path to language model",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="",
        help="prefix before sequence generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.0,
        help="sampling temperature, if temperature == 0, always takes most likely token - greedy decoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=100,
        help="max number of generated words",
    )
    args = parser.parse_args()
    return args
