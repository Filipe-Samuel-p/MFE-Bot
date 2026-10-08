"""
RGCN de PROPAGAÇÃO ÚNICA para tornar a BotRGCN explicável pelo GNNExplainer.

Problema: a `RGCNConv` do PyTorch Geometric propaga uma vez POR TIPO DE RELAÇÃO
(follows / followed_by). O GNNExplainer registra uma máscara de arestas global
(tamanho E) e, ao entrar em cada propagação por relação (subconjunto das arestas),
dispara `assert inputs.size(node_dim) == edge_mask.size(0)` — os tamanhos não batem.

Solução: `FlatRGCNConv` reproduz exatamente a RGCNConv (`aggr='mean'`, com peso
raiz), mas numa ÚNICA passagem sobre TODAS as arestas — assim a máscara de arestas
do explainer casa com o número de mensagens. Os pesos (`weight`, `root`, `bias`)
são reaproveitados do modelo já treinado; nada é retreinado.

Equivalência: como o peso da relação é linear, a média por relação seguida do peso
é igual à soma, por aresta, de (x_j @ weight[r]) / deg_r(dst). É isso que a camada
calcula, com agregação por soma e normalização por aresta pré-computada.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class FlatRGCNConv(MessagePassing):
    def __init__(self, weight, root, bias, num_relations):
        super().__init__(aggr="add")            # somamos; a média vira norm por aresta
        self.register_buffer("weight", weight)  # [R, in, out]
        self.register_buffer("root", root)      # [in, out]
        self.register_buffer("bias", bias)      # [out]
        self.num_relations = num_relations
        self.out_channels = weight.size(2)

    def _edge_norm(self, edge_index, edge_type, num_nodes, device):
        """1 / deg_r(dst) por aresta — replica a média por relação da RGCNConv."""
        dst = edge_index[1]
        norm = torch.zeros(edge_index.size(1), device=device)
        for r in range(self.num_relations):
            m = edge_type == r
            if m.any():
                deg = torch.zeros(num_nodes, device=device)
                deg.index_add_(0, dst[m], torch.ones(int(m.sum()), device=device))
                norm[m] = 1.0 / deg[dst[m]].clamp(min=1.0)
        return norm

    def forward(self, x, edge_index, edge_type):
        norm = self._edge_norm(edge_index, edge_type, x.size(0), x.device)
        agg = self.propagate(edge_index, x=x, edge_type=edge_type, norm=norm)
        return agg + x @ self.root + self.bias

    def message(self, x_j, edge_type, norm):
        out = x_j.new_zeros(x_j.size(0), self.out_channels)
        for r in range(self.num_relations):
            m = edge_type == r
            if m.any():
                out[m] = x_j[m] @ self.weight[r]
        return out * norm.view(-1, 1)


class ExplainBotRGCN(torch.nn.Module):
    """Espelho da BotRGCN treinada usando FlatRGCNConv nas duas camadas.

    Reaproveita os encoders lineares e os pesos das convoluções do modelo base;
    serve APENAS para o GNNExplainer (dropout desligado, sem retreino).
    """

    def __init__(self, base):
        super().__init__()
        self.lin_des = base.lin_des
        self.lin_tweet = base.lin_tweet
        self.lin_num = base.lin_num
        self.lin_cat = base.lin_cat
        self.lin_in = base.lin_in
        self.lin_out = base.lin_out
        r = base.conv1.num_relations
        self.conv1 = FlatRGCNConv(
            base.conv1.weight.detach(), base.conv1.root.detach(),
            base.conv1.bias.detach(), r,
        )
        self.conv2 = FlatRGCNConv(
            base.conv2.weight.detach(), base.conv2.root.detach(),
            base.conv2.bias.detach(), r,
        )

    def encode_nodes(self, des, tweet, num, cat):
        d = F.leaky_relu(self.lin_des(des))
        t = F.leaky_relu(self.lin_tweet(tweet))
        n = F.leaky_relu(self.lin_num(num))
        c = F.leaky_relu(self.lin_cat(cat))
        x = torch.cat([d, t, n, c], dim=1)
        return F.leaky_relu(self.lin_in(x))

    def forward(self, des, tweet, num, cat, edge_index, edge_type):
        x = self.encode_nodes(des, tweet, num, cat)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return self.lin_out(x)
