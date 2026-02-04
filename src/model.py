"""FT-Transformer-style model for tabular data."""

from typing import List

import torch
from torch import nn


def _embed_dim(cardinality: int) -> int:
    return min(64, max(4, int(cardinality ** 0.5) + 1))


class FTTransformerModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        mlp_hidden: int = 256,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.cat_cardinalities = cat_cardinalities

        self.num_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_features)]
        )
        self.num_missing = nn.Parameter(torch.zeros(num_features, d_model))

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)

        mlp = []
        in_dim = d_model
        for _ in range(mlp_layers - 1):
            mlp.extend([nn.Linear(in_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout)])
            in_dim = mlp_hidden
        mlp.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*mlp)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, x_num: torch.Tensor, x_num_mask: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        tokens = []

        if self.num_features > 0:
            for i in range(self.num_features):
                proj = self.num_projections[i](x_num[:, i : i + 1])
                mask = x_num_mask[:, i : i + 1]
                token = proj * mask + self.num_missing[i] * (1.0 - mask)
                tokens.append(token)

        if self.cat_cardinalities:
            for i, emb in enumerate(self.cat_embeddings):
                tokens.append(emb(x_cat[:, i]))

        if tokens:
            x = torch.stack(tokens, dim=1)
        else:
            raise ValueError("No features provided to the model.")

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        x = self.dropout(x[:, 0])
        return self.head(x)
