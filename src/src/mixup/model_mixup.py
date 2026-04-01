# model_mixup.py (mixup-only, same network as baseline)
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BertTextClassifierMixup(nn.Module):
    """
    Mixup 전용 텍스트 모델.
    - baseline BertTextClassifier와 네트워크 구조 동일:
      BERT -> CLS -> Dropout -> Linear(num_labels)
    - 차이: encode/classify를 노출해서 representation-level mixup 지원
    """
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, H)  baseline과 동일
        return cls

    def classify(self, cls: torch.Tensor) -> torch.Tensor:
        x = self.dropout(cls)                 # baseline과 동일
        logits = self.classifier(x)           # baseline과 동일
        return logits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.classify(cls)


class BertTextNumClassifierMixup(nn.Module):
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

        # 수치형 프로젝션: Mixup 전에는 특징 추출에만 집중 (Dropout 제거)
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, hidden_num),
            nn.ReLU(),
            # nn.Dropout(dropout) <- 여기서 제거
        )

        # 공통 분류기: Mixup 및 Concat이 끝난 후 분류기 통과
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden + hidden_num, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS token

    def encode_num(self, num: torch.Tensor) -> torch.Tensor:
        return self.num_proj(num)

    def classify(self, cls_feat: torch.Tensor, num_feat: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([cls_feat, num_feat], dim=1) 
        return self.classifier(feat)

    def forward(self, input_ids, attention_mask, num):
        cls_feat = self.encode(input_ids, attention_mask)
        num_feat = self.encode_num(num)
        return self.classify(cls_feat, num_feat)