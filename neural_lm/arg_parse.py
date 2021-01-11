from argparse import ArgumentParser, Namespace


def get_train_args() -> Namespace:
    """
    Training Argument Parser.

    :return: parsed arguments
    :rtype: Namespace
    """

    parser = ArgumentParser()

    # path
    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="path to train data",
    )
    parser.add_argument(
        "--path_to_save_folder",
        type=str,
        required=False,
        default="models/rnn_language_model",
        help="path to save folder",
    )

    # dataset and dataloader
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=None,
        help="max sentence length (chars)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="dataloader batch_size",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        required=False,
        default=True,
        help="dataloader shuffle",
    )

    # model
    parser.add_argument(
        "--embedding_dim",
        type=int,
        required=True,
        help="embedding dimension",
    )
    parser.add_argument(
        "--rnn_hidden_size",
        type=int,
        required=True,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--rnn_num_layers",
        type=int,
        required=False,
        default=1,
        help="number of LSTM layers",
    )
    parser.add_argument(
        "--rnn_dropout",
        type=float,
        required=False,
        default=0.0,
        help="LSTM dropout",
    )

    # train
    parser.add_argument(
        "--n_epoch",
        type=int,
        required=True,
        help="number of epochs",
    )
    parser.add_argument(
        "--train_eval_freq",
        type=int,
        required=False,
        default=50,
        help="evaluation frequency (number of batches)",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        required=False,
        default=1.0,
        help="max_norm parameter in clip_grad_norm",
    )

    # additional
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        choices=["cpu", "cuda"],
        help="torch device (available: 'cpu', 'cuda')",
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
        "--path_to_model_folder",
        type=str,
        required=True,
        help="path to language model folder",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=None,
        help="max sentence length (chars)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        choices=["cpu", "cuda"],
        help="torch device (available: 'cpu', 'cuda')",
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
        "--path_to_model_folder",
        type=str,
        required=True,
        help="path to language model folder",
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
        help="max number of generated tokens (chars)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        choices=["cpu", "cuda"],
        help="torch device (available: 'cpu', 'cuda')",
    )

    args = parser.parse_args()
    return args
