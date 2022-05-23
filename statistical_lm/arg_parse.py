from argparse import ArgumentParser, Namespace


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
        "--path_to_save",
        type=str,
        required=False,
        default="models/language_model.pkl",
        help="path to save model",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="n-gram order",
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
        help="path to validation data",
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
        "--strategy",
        type=str,
        required=False,
        default="sampling",
        choices=["sampling", "top-k", "top-p", "beam search"],
        help="sampling strategy (available: 'sampling', 'top-k', 'top-p' and 'beam search')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.0,
        help="sampling temperature, if temperature == 0, always takes most likely token - greedy decoding",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=10,
        help="top-k parameter (only for 'top-k' sampling strategy)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=100,
        help="max number of generated words",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()
    return args
