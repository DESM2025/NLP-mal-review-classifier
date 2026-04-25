from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertModel


class DistilBertBiLSTMClassifier(nn.Module):
    """Clasificador binario con DistilBERT como extractor y BiLSTM como cabeza."""

    def __init__(
        self,
        pretrained_model_name: str = "distilbert-base-uncased",
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        dropout: float = 0.30,
        freeze_distilbert: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)

        hidden_size = int(self.distilbert.config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, 1)

        if freeze_distilbert:
            self.freeze_distilbert()

    def freeze_distilbert(self) -> None:
        """Congela DistilBERT para entrenar solo la parte recurrente/clasificadora."""
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_distilbert(self) -> None:
        """Descongela DistilBERT para fine-tuning end-to-end."""
        for param in self.distilbert.parameters():
            param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        lengths = attention_mask.sum(dim=1).clamp(min=1).to(torch.int64).cpu()
        packed = pack_padded_sequence(
            sequence_output,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)

        h_forward = h_n[-2]
        h_backward = h_n[-1]
        features = torch.cat((h_forward, h_backward), dim=1)

        logits = self.classifier(self.dropout(features)).squeeze(-1)
        return logits
