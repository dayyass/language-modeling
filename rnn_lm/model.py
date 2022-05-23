import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNLanguageModel(nn.Module):
    """
    RNN Neural Network for Language Modeling.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rnn_hidden_size: int,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
    ):
        """
        Init model with nn.Embedding, nn.LSTM and nn.Linear.

        :param int num_embeddings: vocabulary size
        :param int embedding_dim: embedding dimension
        :param int rnn_hidden_size: LSTM hidden size
        :param int rnn_num_layers: number of LSTM layers (default: 1)
        :param float rnn_dropout: LSTM dropout (default: 0.0)
        """

        super().__init__()

        self.emb = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=False,  # causal language model
        )
        self.logits = nn.Linear(rnn_hidden_size, num_embeddings)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute logits given input tokens.

        :param torch.Tensor x: batch of sequences with token indices
        :param torch.Tensor lengths: length of each sentence in batch
        :return: pre-softmax linear outputs of language model
        :rtype: torch.Tensor
        """

        # embedding
        emb = self.emb(x)

        # rnn
        packed_emb = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_rnn, _ = self.rnn(packed_emb)
        rnn, _ = pad_packed_sequence(
            packed_rnn, batch_first=True, padding_value=0.0  # hardcoded
        )

        # linear
        logits = self.logits(rnn)

        return logits
