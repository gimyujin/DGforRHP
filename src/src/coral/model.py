#coral/model.py

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel

Tensor = torch.Tensor

class BertTextCoralClassifier(nn.Module):
    """
    Text-only CORAL model (ERM과 동일한 1-layer classifier 구조 유지)
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_features: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, H)
        
        # ERM 구조와 동일하게 BERT 직후가 아닌 분류기 직전에 드롭아웃 적용
        x = self.dropout(cls)
        logits = self.classifier(x)

        if return_features:
            # Text-only의 경우 정렬할 피처로 드롭아웃 전의 CLS 혹은 드롭아웃 후의 x를 사용
            return logits, x
        return logits


class BertTextNumCoralClassifier(nn.Module):
    """
    Text + numeric CORAL model (ERM과 구조를 완벽히 일치시킴)
    
    ERM 구조:
      - num_proj: num_features -> hidden_num -> ReLU -> Dropout
      - concat: bert_hidden + hidden_num
      - classifier: (Linear 256) -> ReLU -> Dropout -> (Linear labels)
    """
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

        # 1. ERM과 동일한 Numeric Projection
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, hidden_num),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. ERM과 동일한 Classifier 구조 (3단 구성)
        # CORAL 피처 추출을 위해 Sequential을 풀어서 정의
        self.fc1 = nn.Linear(bert_hidden + hidden_num, 256)
        self.act = nn.ReLU()
        self.fc_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        num: Optional[Tensor] = None,
        return_features: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if num is None:
            raise ValueError("num input is required for TextNum model.")

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768) - 드롭아웃 없음 (ERM 일치)

        num_h = self.num_proj(num)            # (B, hidden_num)
        feat = torch.cat([cls, num_h], dim=1) # (B, 768+hidden_num)

        # Classifier Path
        h = self.fc1(feat)
        h = self.act(h)
        h = self.fc_drop(h)  # <--- 이 지점(h)이 ERM의 '256 레이어' 결과물이며 CORAL의 정렬 대상

        logits = self.fc2(h)

        if return_features:
            return logits, h
        return logits