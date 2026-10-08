"""
Scoring dos modelos-base sobre o grafo do TwiBot-20.

Módulo COMPARTILHADO entre o passo 6 (training/train_lr.py) e o worker
(worker/run_inference.py). Centraliza o carregamento dos modelos serializados
e a geração de p_rf / p_gnn, garantindo que o meta-classificador seja treinado
exatamente com os scores que o worker produzirá em produção.

Observações:
- O RF usa as features CRUS (X_raw já presentes no grafo), pois o próprio
  modelo calibrado encapsula qualquer transformação interna.
- O GNN é transdutivo: uma única passada no grafo inteiro devolve p_gnn para
  TODOS os nós de uma vez (não há scoring por-usuário isolado).
"""

import joblib
import numpy as np
import torch

from botrgcn import BotRGCN, normalize
from features import N_NUMERIC


def load_rf(path):
    """Carrega o RandomForest calibrado (CalibratedClassifierCV)."""
    return joblib.load(path)


def rf_scores(rf_model, X_raw):
    """p_rf = probabilidade calibrada da classe 'bot' para cada nó."""
    return rf_model.predict_proba(X_raw)[:, 1]


def load_gnn(path):
    """Carrega o bundle do GNN e reconstrói o modelo multimodal em modo eval."""
    bundle = torch.load(path, weights_only=False)
    model = BotRGCN(
        des_dim=bundle["des_dim"],
        tweet_dim=bundle["tweet_dim"],
        num_dim=bundle["num_dim"],
        cat_dim=bundle["cat_dim"],
        hidden_dim=bundle["hidden_dim"],
        num_relations=bundle["num_relations"],
        dropout=bundle["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


@torch.no_grad()
def gnn_scores(gnn_model, bundle, graph):
    """p_gnn para todos os nós, via uma única passada no grafo transdutivo."""
    X_num = normalize(
        graph["X_raw"][:, :N_NUMERIC], bundle["norm_mean"], bundle["norm_std"]
    )
    des = torch.tensor(graph["des_emb"], dtype=torch.float32)
    tweet = torch.tensor(graph["tweet_emb"], dtype=torch.float32)
    num = torch.tensor(X_num, dtype=torch.float32)
    cat = torch.tensor(graph["X_raw"][:, N_NUMERIC:], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    edge_type = torch.tensor(graph["edge_type"], dtype=torch.long)
    logits = gnn_model(des, tweet, num, cat, edge_index, edge_type)
    proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return np.asarray(proba, dtype=np.float64)
