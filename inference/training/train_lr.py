"""
Fase 1 — Passo 6 — Treino da Regressão Logística (Camada de Arbitragem).

Aqui o dev.json finalmente entra em cena. Os dois modelos-base CONGELADOS
(RF + BotRGCN) pontuam o dev; cada usuário vira um par (p_rf, p_gnn). A LR é
treinada APENAS sobre os casos de DIVERGÊNCIA de classe (pred_rf != pred_gnn),
que é o único regime onde ela é acionada no roteamento (CLAUDE.md, camada 3).

Por que só divergência:
- No roteamento, casos convergentes usam a média dos scores (Early Exit) e
  NUNCA chamam a LR. Treinar a LR no regime em que ela opera evita mismatch
  train/serve e torna seus coeficientes diretamente interpretáveis como o
  "peso histórico" de cada modelo quando os dois discordam (modelo caixa-branca).

Regras respeitadas:
- Meta-classificador treinado apenas com scores do dev.json (regra 5).
- Usuários com neighbor:null não entram no stacking (regra 4) -> excluídos.

Uso:
    python inference/training/train_lr.py
"""

import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)

from graph import build_graph  # noqa: E402
from scoring import gnn_scores, load_gnn, load_rf, rf_scores  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(INFERENCE_DIR, "models")
RF_PATH = os.path.join(MODELS_DIR, "modelo_rf.pkl")
GNN_PATH = os.path.join(MODELS_DIR, "modelo_gnn.pt")
LR_PATH = os.path.join(MODELS_DIR, "modelo_lr.pkl")

SEED = 42
SPLIT_DEV = 1


def main():
    print("=" * 60)
    print("Treino da Regressão Logística (meta-classificador) — MFE-Bot")
    print("=" * 60)

    print("\nConstruindo grafo e carregando modelos congelados...")
    g = build_graph(DATA_DIR)
    rf = load_rf(RF_PATH)
    gnn, bundle = load_gnn(GNN_PATH)

    print("Pontuando dev.json com RF e BotRGCN...")
    p_rf_all = rf_scores(rf, g["X_raw"])
    p_gnn_all = gnn_scores(gnn, bundle, g)

    # Seleciona usuários do dev que ENTRAM no stacking (neighbor != null).
    is_dev = g["split"] == SPLIT_DEV
    in_stack = is_dev & g["has_neighbor"]
    idx = np.where(in_stack)[0]

    p_rf = p_rf_all[idx]
    p_gnn = p_gnn_all[idx]
    y = g["y"][idx]

    pred_rf = (p_rf >= 0.5).astype(int)
    pred_gnn = (p_gnn >= 0.5).astype(int)
    divergente = pred_rf != pred_gnn

    n_dev = int(is_dev.sum())
    n_stack = len(idx)
    n_div = int(divergente.sum())
    print("\n--- Diagnóstico do dev ---")
    print(f"  usuários no dev:                 {n_dev}")
    print(f"  com neighbor (entram no stacking): {n_stack}")
    print(f"  neighbor:null (RF-only, excluídos): {n_dev - n_stack}")
    print(f"  CONVERGENTES: {n_stack - n_div}  |  DIVERGENTES: {n_div} "
          f"({100*n_div/max(n_stack,1):.1f}%)")

    Xd = np.column_stack([p_rf[divergente], p_gnn[divergente]])
    yd = y[divergente]
    bots = int((yd == 1).sum())
    print(f"  divergentes -> bots={bots} humanos={len(yd)-bots}")

    if len(yd) < 20 or bots == 0 or bots == len(yd):
        print("\n[ATENÇÃO] Poucos casos divergentes ou uma única classe. "
              "A LR pode ficar instável — reavaliar estratégia se for o caso.")

    # LR caixa-branca sobre [p_rf, p_gnn].
    lr = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)

    # Estimativa honesta de desempenho da LR via validação cruzada no
    # próprio conjunto divergente (sem holdout extra — eval final é no test).
    n_splits = min(5, bots, len(yd) - bots)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        yhat = cross_val_predict(lr, Xd, yd, cv=skf)
        print(f"\n  F1 (CV {n_splits}-fold) da LR nos divergentes: "
              f"{f1_score(yd, yhat, zero_division=0):.4f}")

    # Baselines no MESMO conjunto divergente, para comparar.
    print("  baseline 'sempre RF'  F1:",
          f"{f1_score(yd, pred_rf[divergente], zero_division=0):.4f}")
    print("  baseline 'sempre GNN' F1:",
          f"{f1_score(yd, pred_gnn[divergente], zero_division=0):.4f}")

    # Ajuste final em TODOS os divergentes do dev e serialização.
    lr.fit(Xd, yd)
    coef = lr.coef_[0]
    print("\n--- LR (caixa-branca) ---")
    print(f"  coef p_rf  = {coef[0]:+.4f}")
    print(f"  coef p_gnn = {coef[1]:+.4f}")
    print(f"  intercepto = {lr.intercept_[0]:+.4f}")
    print("  (maior coeficiente = modelo com mais peso histórico na discordância)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr, LR_PATH)
    print(f"\nModelo salvo em: {LR_PATH}")


if __name__ == "__main__":
    main()
