# model.py (text+num)
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

class BertTextClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT의 pooled output이 없는 모델도 있어서 CLS 토큰으로 통일
        cls = out.last_hidden_state[:, 0, :]  # (B, H)
        x = self.dropout(cls)
        logits = self.classifier(x)           # (B, 2)
        return logits


class BertTextNumClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_features: int,
        num_labels: int = 2,
        dropout: float = 0.1,
        hidden_num: int = 128,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size

        self.num_proj = nn.Sequential(
            nn.Linear(num_features, hidden_num),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden + hidden_num, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, num):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768)

        num_h = self.num_proj(num)            # (B, hidden_num)
        feat = torch.cat([cls, num_h], dim=1) # (B, 768+hidden_num)

        logits = self.classifier(feat)
        return logits
