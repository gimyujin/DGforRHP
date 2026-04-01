from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import AutoModel

# =========================
# GRL (Gradient Reversal Layer) - 유지
# =========================
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None

def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)

@dataclass
class DANNOutputs:
    y_logits: torch.Tensor  # (B, num_labels)
    d_logits: torch.Tensor  # (B, num_domains)

class DANNModel(nn.Module):
    """
    DANN (DG mode) - ERM 구조와 완벽 일치 버전
    """
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        num_domains: int,
        use_numeric: bool = False,
        num_numeric: int = 0,
        dropout: float = 0.1,
        hidden_num: int = 128,
        head_hidden: int = 256, # ERM의 256 레이어와 매칭
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        bert_hidden = self.encoder.config.hidden_size
        self.use_numeric = use_numeric

        # 1. ERM과 동일한 Numeric Projection
        if self.use_numeric:
            self.num_proj = nn.Sequential(
                nn.Linear(num_numeric, hidden_num),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            feat_dim = bert_hidden + hidden_num
        else:
            self.num_proj = None
            feat_dim = bert_hidden

        # 2. Label Head (ERM의 classifier 구조와 1:1 매칭)
        self.label_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_labels),
        )

        # 3. Domain Head (Label Head와 동일한 깊이 유지)
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_domains),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
    ) -> DANNOutputs:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, H) - BERT 직후 드롭아웃 제거 (ERM 일치)

        if self.use_numeric:
            if num is None:
                raise ValueError("use_numeric=True but num is missing.")
            num_h = self.num_proj(num)
            feat = torch.cat([cls, num_h], dim=1) # (B, H + hidden_num)
        else:
            feat = cls

        # Label 예측 (Main Task)
        y_logits = self.label_head(feat)

        # Domain 예측 (Adversarial Task via GRL)
        rev_feat = grad_reverse(feat, grl_lambda)
        d_logits = self.domain_head(rev_feat)

        return DANNOutputs(y_logits=y_logits, d_logits=d_logits)