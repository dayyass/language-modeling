from collections import defaultdict
from typing import Callable, DefaultDict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import infer_lengths, masking

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
    :param float clip_grad_norm: max_norm param in clip_grad_norm (default: 1.0)
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
    :param float clip_grad_norm: max_norm param in clip_grad_norm (default: 1.0)
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
