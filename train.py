import json
import os
from collections import defaultdict
from typing import Callable, DefaultDict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import RNNLanguageModel
from utils import (
    LMCollator,
    LMDataset,
    get_char2idx,
    get_train_args,
    infer_lengths,
    load_data,
    masking,
    set_global_seed,
)

BOS = "<BOS>"  # hardcoded
EOS = "<EOS>"  # hardcoded


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    train_eval_freq: int = 50,
    clip_grad_norm: float = 1.0,
    verbose: bool = True,
) -> DefaultDict[str, List[float]]:
    """
    Training loop on one epoch.

    :param nn.Module model: PyTorch Neural Network
    :param DataLoader dataloader: PyTorch DataLoader
    :param Callable criterion: PyTorch Critertion
    :param optim.Optimizer optimizer: PyTorch Optimizer
    :param torch.device device: PyTorch Device
    :param int train_eval_freq: evaluation frequency (number of batches) (default: 50)
    :param float clip_grad_norm: max_norm parameter in clip_grad_norm (default: 1.0)
    :param bool verbose: verbose (default: True)
    :return: metrics dict
    :rtype: DefaultDict[str, List[float]]
    """

    metrics: DefaultDict[str, List[float]] = defaultdict(list)

    # BOS and EOS
    bos_id = dataloader.dataset.char2idx[BOS]
    eos_id = dataloader.dataset.char2idx[EOS]

    if verbose:
        dataloader = tqdm(dataloader, desc="iter dataloader")

    model.train()

    for i, sentence in enumerate(dataloader):
        sentence = sentence.to(device)

        # lengths and mask
        targets = sentence[:, 1:]  # clip left
        lengths = infer_lengths(sentence, bos_id=bos_id, eos_id=eos_id)
        mask = masking(lengths + 1)  # incl. EOS

        # forward pass
        outputs = model(
            sentence[:, :-1],  # clip right
            lengths + 1,  # incl. BOS
        )
        loss_matrix = criterion(
            input=outputs.transpose(1, 2),
            target=targets,
        )
        loss = (loss_matrix * mask).sum() / mask.sum()

        # backward pass
        loss.backward()

        # clip grad norm
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_grad_norm,
        )

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # calculate metrics
        metrics["loss"].append(loss.item())
        metrics["grad_norm"].append(grad_norm.item())

        if verbose:
            if i % train_eval_freq == 0:
                for metric_name, metric_list in metrics.items():
                    print(f"{metric_name}: {np.mean(metric_list[-train_eval_freq:])}")
                print()

    return metrics


def train(
    n_epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    train_eval_freq: int = 50,
    clip_grad_norm: float = 1.0,
    verbose: bool = True,
):
    """
    Training loop for n_epoch.

    :param n_epoch: number of epochs
    :param nn.Module model: PyTorch Neural Network
    :param DataLoader dataloader: PyTorch DataLoader
    :param Callable criterion: PyTorch Critertion
    :param optim.Optimizer optimizer: PyTorch Optimizer
    :param torch.device device: PyTorch Device
    :param int train_eval_freq: evaluation frequency (number of batches) (default: 50)
    :param float clip_grad_norm: max_norm parameter in clip_grad_norm (default: 1.0)
    :param bool verbose: verbose (default: True)
    """

    epochs_range = range(n_epoch)
    if verbose:
        epochs_range = tqdm(epochs_range, desc="iter epochs")

    for _ in epochs_range:

        metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_eval_freq=train_eval_freq,
            clip_grad_norm=clip_grad_norm,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in metrics.items():
                print(f"train {metric_name}: {np.mean(metric_list)}")
            print()


if __name__ == "__main__":

    # argparse
    args = get_train_args()

    # check path_to_save existence
    if os.path.exists(args.path_to_save_folder):
        raise FileExistsError("save path folder already exists")

    # set seed and device
    set_global_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = load_data(path=args.path_to_data, verbose=args.verbose)

    # char2idx
    char2idx = get_char2idx(data, BOS=BOS, EOS=EOS, verbose=args.verbose)

    # dataset, collator, dataloader
    train_dataset = LMDataset(
        data,
        char2idx,
        max_len=args.max_len,
        BOS=BOS,
        EOS=EOS,
        verbose=args.verbose,
    )
    train_collator = LMCollator(
        padding_value=char2idx[EOS],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=train_collator,
    )

    # model
    model = RNNLanguageModel(
        num_embeddings=len(char2idx),
        embedding_dim=args.embedding_dim,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        rnn_dropout=args.rnn_dropout,
    ).to(device)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")  # use mask for reduction
    optimizer = optim.Adam(model.parameters())

    # train
    train_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_eval_freq=args.train_eval_freq,
        clip_grad_norm=args.clip_grad_norm,
        verbose=args.verbose,
    )

    # save
    os.makedirs(args.path_to_save_folder, exist_ok=True)
    # # model
    torch.save(
        model.eval().cpu().state_dict(),
        os.path.join(args.path_to_save_folder, "language_model.pth"),
    )
    # # vocab char2idx
    with open(os.path.join(args.path_to_save_folder, "vocab.json"), "w") as fp:
        json.dump(char2idx, fp)
