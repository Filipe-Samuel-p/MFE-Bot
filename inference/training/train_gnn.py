"""
Fase 1 — Treino da BotRGCN multimodal (Camada Base, especialista de REDE).

Grafo transdutivo único (graph.py) com todos os usuários rotulados. Cada nó
funde 4 modalidades (description + tweets via RoBERTa, props numéricas e
categóricas de perfil), sobre as quais roda uma RGCN de 2 relações direcionadas
(follows / followed_by). Fiel a Feng et al. (2021a).

Princípios metodológicos:
- TRANSDUTIVO: features e arestas de dev/test entram no grafo, mas seus LABELS
  nunca entram na loss (treina só com a máscara de treino).
- dev.json fica INTOCADO: o early-stopping usa um holdout interno carved do
  treino. O dev só é usado no passo 6 (Regressão Logística).

Pré-requisito: rodar antes `training/encode_text.py` (gera models/text_emb.npz).

Uso:
    python inference/training/train_gnn.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)

from botrgcn import BotRGCN, normalize  # noqa: E402
from features import N_NUMERIC  # noqa: E402
from graph import NUM_RELATIONS, build_graph  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(INFERENCE_DIR, "models", "modelo_gnn.pt")

SEED = 42
HIDDEN_DIM = 128  # divisível por 4 (D/4 por modalidade)
DROPOUT = 0.3
LR = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
PATIENCE = 30
VAL_FRAC = 0.15  # holdout interno carved do treino (dev fica intocado)


def evaluate(logits, y, mask, nome):
    pred = logits.argmax(dim=1).cpu().numpy()[mask]
    yt = y.cpu().numpy()[mask]
    print(f"\n[{nome}]")
    print(f"  Acurácia : {accuracy_score(yt, pred):.4f}")
    print(f"  Precisão : {precision_score(yt, pred, zero_division=0):.4f}")
    print(f"  Revocação: {recall_score(yt, pred, zero_division=0):.4f}")
    f1 = f1_score(yt, pred, zero_division=0)
    print(f"  F1-Score : {f1:.4f}")
    return f1


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Treino da BotRGCN multimodal (transdutivo) — MFE-Bot")
    print("=" * 60)

    print("\nConstruindo grafo...")
    g = build_graph(DATA_DIR)
    if g["des_emb"] is None:
        sys.exit(
            "ERRO: embeddings de texto ausentes. Rode primeiro:\n"
            "  python inference/training/encode_text.py"
        )
    N = len(g["ids"])
    E = g["edge_index"].shape[1]
    split = g["split"]
    print(f"  nós: {N} | arestas (direcionadas): {E}")
    print(
        f"  train={int((split==0).sum())} dev={int((split==1).sum())} "
        f"test={int((split==2).sum())} support={int((split==3).sum())}"
    )
    isolated = N - len(np.unique(g["edge_index"]))
    print(f"  nós isolados: {isolated}")

    # Separa modalidades. Numéricas: log1p + z-score (ajuste só no treino).
    X_num_raw = g["X_raw"][:, :N_NUMERIC]
    X_cat = g["X_raw"][:, N_NUMERIC:]
    train_idx_all = np.where(split == 0)[0]
    x_log_train = np.log1p(np.clip(X_num_raw[train_idx_all], 0.0, None))
    mean = x_log_train.mean(axis=0)
    std = x_log_train.std(axis=0) + 1e-6
    X_num = normalize(X_num_raw, mean, std)

    des = torch.tensor(g["des_emb"], dtype=torch.float32)
    tweet = torch.tensor(g["tweet_emb"], dtype=torch.float32)
    num = torch.tensor(X_num, dtype=torch.float32)
    cat = torch.tensor(X_cat, dtype=torch.float32)
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_type = torch.tensor(g["edge_type"], dtype=torch.long)
    y = torch.tensor(np.clip(g["y"], 0, None), dtype=torch.long)

    # Holdout interno carved do treino para early-stopping (dev intocado).
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(train_idx_all)
    n_val = int(len(perm) * VAL_FRAC)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[tr_idx] = True

    counts = np.bincount(g["y"][tr_idx], minlength=2).astype(np.float64)
    class_w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)

    model = BotRGCN(
        des_dim=des.shape[1],
        tweet_dim=tweet.shape[1],
        num_dim=num.shape[1],
        cat_dim=cat.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_relations=NUM_RELATIONS,
        dropout=DROPOUT,
    )
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def fwd():
        return model(des, tweet, num, cat, edge_index, edge_type)

    print("\nTreinando...")
    best_f1, best_state, since_best = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optim.zero_grad()
        logits = fwd()
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_w)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            pred = fwd().argmax(dim=1).cpu().numpy()
            vf1 = f1_score(g["y"][val_idx], pred[val_idx], zero_division=0)
        if vf1 > best_f1:
            best_f1, since_best = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if epoch % 20 == 0 or since_best == 0:
            print(f"  época {epoch:3d} | loss {loss.item():.4f} | val F1 {vf1:.4f}")
        if since_best >= PATIENCE:
            print(f"  early stop na época {epoch} (melhor val F1 {best_f1:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = fwd()
    evaluate(logits, y, tr_idx, "TREINO interno")
    evaluate(logits, y, val_idx, "VAL interno (holdout)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "des_dim": int(des.shape[1]),
            "tweet_dim": int(tweet.shape[1]),
            "num_dim": int(num.shape[1]),
            "cat_dim": int(cat.shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "num_relations": NUM_RELATIONS,
            "dropout": DROPOUT,
            "norm_mean": mean,
            "norm_std": std,
            "feature_names": g["feature_names"],
        },
        MODEL_PATH,
    )
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
