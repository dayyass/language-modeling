from typing import List

from tqdm import tqdm


def load_data(path: str, verbose: bool = True) -> List[str]:
    """
    Load data.

    :param str path: path to load data
    :param bool verbose: verbose (default: True)
    :return: data
    :rtype: List[str]
    """

    data = []
    with open(path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp, desc="load data")
        for line in fp:
            data.append(line)
    return data
