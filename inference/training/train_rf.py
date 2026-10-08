"""
Fase 1 — Treino da Random Forest (Camada Base, especialista de PERFIL).

Lê o TwiBot-20 em streaming (ijson), extrai as features tabulares de perfil
(features.py) e treina uma RandomForest ENVOLVIDA em CalibratedClassifierCV
(método isotônico, cv=5).

A calibração é obrigatória (CLAUDE.md, regra 7): o roteador por cruzamento de
fronteira decide em p_rf >= 0.5, então a probabilidade precisa ser calibrada
para que esse limiar represente uma classificação real, e não um artefato do
algoritmo.

Avalia no dev.json e serializa o modelo calibrado em models/modelo_rf.pkl.

Uso:
    python inference/training/train_rf.py
"""

import os
import sys
import time

import ijson
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Permite importar o módulo compartilhado de features (inference/features.py)
INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)

from features import FEATURE_NAMES, extract_features, parse_label  # noqa: E402

# IMPORTANTE: o dev.json NÃO é usado aqui. Ele é reservado para o passo 6
# (gerar scores que treinam a Regressão Logística), depois que RF e GNN estão
# treinados e congelados. Para o sanity check de generalização usamos um
# holdout interno do próprio train.json, mantendo o dev intocado.
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "train.json")
MODEL_PATH = os.path.join(INFERENCE_DIR, "models", "modelo_rf.pkl")

RANDOM_STATE = 42
HOLDOUT_FRAC = 0.2


def load_dataset(path):
    """Lê um arquivo do TwiBot-20 em streaming e devolve (X, y) como arrays.

    Usuários sem label são ignorados (não servem para treino/avaliação).
    """
    X, y = [], []
    t0 = time.time()
    with open(path, "rb") as fh:
        for user in ijson.items(fh, "item"):
            label = parse_label(user)
            if label is None:
                continue
            X.append(extract_features(user))
            y.append(label)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    print(
        f"  {os.path.basename(path)}: {len(y)} usuários "
        f"({int(y.sum())} bots / {int((y == 0).sum())} humanos) "
        f"em {time.time() - t0:.1f}s"
    )
    return X, y


def evaluate(model, X, y, nome):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n[{nome}]")
    print(f"  Acurácia : {accuracy_score(y, pred):.4f}")
    print(f"  Precisão : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"  Revocação: {recall_score(y, pred, zero_division=0):.4f}")
    print(f"  F1-Score : {f1_score(y, pred, zero_division=0):.4f}")
    print(f"  Brier    : {brier_score_loss(y, proba):.4f}  (menor = melhor calibração)")


def main():
    print("=" * 60)
    print("Treino da Random Forest (calibrada) — MFE-Bot")
    print("=" * 60)

    print("\nCarregando dados...")
    X, y = load_dataset(TRAIN_PATH)
    print(f"  features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    def build_model():
        # RF base. class_weight='balanced' compensa eventual desbalanceamento.
        base_rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        # Envoltório de calibração — OBRIGATÓRIO (CLAUDE.md regra 7).
        # cv=5 treina a RF internamente em folds e calibra com isotônica.
        return CalibratedClassifierCV(estimator=base_rf, method="isotonic", cv=5)

    # --- Sanity check: holdout interno do train (dev fica intocado) ---
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=HOLDOUT_FRAC, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"\nSanity check em holdout interno "
        f"({len(y_tr)} treino / {len(y_ho)} holdout)..."
    )
    sanity = build_model()
    sanity.fit(X_tr, y_tr)
    evaluate(sanity, X_tr, y_tr, "TREINO (80%)")
    evaluate(sanity, X_ho, y_ho, "HOLDOUT (20%)")

    # --- Modelo final: treinado em TODO o train.json e serializado ---
    print("\nTreinando modelo final em 100% do train.json...")
    t0 = time.time()
    model = build_model()
    model.fit(X, y)
    print(f"  concluído em {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo calibrado salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
