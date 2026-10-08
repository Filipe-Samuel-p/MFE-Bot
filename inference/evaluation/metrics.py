"""
Fase 3 — Avaliação Quantitativa (prova da tese).

Lê a tabela `auditoria_deteccao_bots` (populada pelo worker) e calcula, para os
três competidores, Acurácia / Precisão / Revocação / F1-Score sobre o test.json:
- RF isolado    -> limiar 0.5 sobre score_rf
- BotRGCN isolado -> limiar 0.5 sobre score_gnn
- MFE-Bot       -> classificacao_bot já gravada (todo o pipeline de roteamento)

Critério de falseabilidade (TCC §3.5.1): a hipótese se sustenta se
F1(MFE-Bot) > F1(RF) e F1(MFE-Bot) > F1(BotRGCN).

Uso:
    python inference/evaluation/metrics.py
"""

import os
import sys

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)

from db import connect  # noqa: E402


def metricas(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )


def main():
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT score_rf, score_gnn, classificacao_bot, label_real "
                "FROM auditoria_deteccao_bots WHERE label_real IS NOT NULL"
            )
            linhas = cur.fetchall()
    finally:
        conn.close()

    if not linhas:
        sys.exit("Tabela vazia. Rode antes: python inference/worker/run_inference.py")

    y_true = [r[3] for r in linhas]
    y_rf = [1 if r[0] >= 0.5 else 0 for r in linhas]
    y_gnn = [1 if r[1] >= 0.5 else 0 for r in linhas]
    y_sys = [r[2] for r in linhas]

    print("=" * 60)
    print(f"Avaliação no test.json — {len(linhas)} usuários")
    print("=" * 60)
    header = f"{'Modelo':<12}{'Acurácia':>10}{'Precisão':>10}{'Revocação':>11}{'F1':>9}"
    print(header)
    print("-" * len(header))
    resultados = {}
    for nome, y_pred in [("RF", y_rf), ("BotRGCN", y_gnn), ("MFE-Bot", y_sys)]:
        acc, prec, rec, f1 = metricas(y_true, y_pred)
        resultados[nome] = f1
        print(f"{nome:<12}{acc:>10.4f}{prec:>10.4f}{rec:>11.4f}{f1:>9.4f}")

    print("-" * len(header))
    f1_sys, f1_rf, f1_gnn = resultados["MFE-Bot"], resultados["RF"], resultados["BotRGCN"]
    if f1_sys > f1_rf and f1_sys > f1_gnn:
        print(f"✓ TESE SUSTENTADA: F1(MFE-Bot)={f1_sys:.4f} > RF={f1_rf:.4f} e GNN={f1_gnn:.4f}")
    else:
        print(f"✗ Tese NÃO sustentada: F1(MFE-Bot)={f1_sys:.4f} vs RF={f1_rf:.4f} / GNN={f1_gnn:.4f}")


if __name__ == "__main__":
    main()
