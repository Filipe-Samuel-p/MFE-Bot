"""
Definição da BotRGCN multimodal e da normalização de features numéricas.

Fiel ao BotRGCN (Feng et al., 2021a): as features iniciais de cada nó fundem
QUATRO modalidades, cada uma projetada para D/4 e concatenadas em D:
  - description (embedding RoBERTa, 768)
  - tweets      (média de embeddings RoBERTa, 768)
  - num_prop    (atributos numéricos de perfil, z-score)
  - cat_prop    (atributos categóricos/booleanos de perfil)

Sobre essa representação inicial roda uma RGCN de 2 camadas com duas relações
direcionadas (follows / followed_by). Módulo COMPARTILHADO entre treino e
scoring para garantir arquitetura idêntica em treino e inferência.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class BotRGCN(torch.nn.Module):
    def __init__(self, des_dim, tweet_dim, num_dim, cat_dim,
                 hidden_dim, num_relations, dropout):
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim deve ser divisível por 4"
        q = hidden_dim // 4

        # Encoders por modalidade -> D/4 cada.
        self.lin_des = torch.nn.Linear(des_dim, q)
        self.lin_tweet = torch.nn.Linear(tweet_dim, q)
        self.lin_num = torch.nn.Linear(num_dim, q)
        self.lin_cat = torch.nn.Linear(cat_dim, q)

        # Fusão + propagação relacional.
        self.lin_in = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.lin_out = torch.nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def encode_nodes(self, des, tweet, num, cat):
        d = F.leaky_relu(self.lin_des(des))
        t = F.leaky_relu(self.lin_tweet(tweet))
        n = F.leaky_relu(self.lin_num(num))
        c = F.leaky_relu(self.lin_cat(cat))
        x = torch.cat([d, t, n, c], dim=1)
        return F.leaky_relu(self.lin_in(x))

    def forward(self, des, tweet, num, cat, edge_index, edge_type):
        x = self.encode_nodes(des, tweet, num, cat)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)


def normalize(X_raw, mean, std):
    """log1p (features numéricas são >= 0) + z-score. Idêntico em treino/serve."""
    x_log = np.log1p(np.clip(X_raw, 0.0, None))
    return (x_log - mean) / std
