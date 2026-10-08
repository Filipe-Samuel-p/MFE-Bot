"""
Fase 3 — Significância estatística dos ganhos de F1 (TCC §5, Tabela de significância).

Complementa metrics.py: além das estimativas pontuais, quantifica a incerteza
dos ganhos do MFE-Bot sobre cada especialista com dois testes:

1. Bootstrap não paramétrico (Efron & Tibshirani, 1994):
   reamostra o conjunto de teste com reposição B vezes (semente fixa) e
   calcula, em cada reamostra, a diferença de F1 (MFE-Bot - especialista).
   Reporta o delta médio, o IC 95% (percentis 2,5/97,5) e P(delta <= 0).

2. Teste exato de McNemar (McNemar, 1947), bicaudal:
   compara os classificadores apenas nos casos discordantes (um acerta e o
   outro erra), via distribuição binomial exata.

Uso:
    python inference/evaluation/significance.py [--reamostras 10000] [--semente 42]
"""

import argparse
import os
import sys
from math import comb

import numpy as np
from sklearn.metrics import f1_score

INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)

from db import connect  # noqa: E402


def bootstrap_delta_f1(y, pred_a, pred_b, n_boot, seed):
    """Distribuição bootstrap de F1(a) - F1(b). Retorna (media, lo, hi, p_le_0)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = f1_score(y[idx], pred_a[idx]) - f1_score(y[idx], pred_b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi), float((deltas <= 0).mean())


def mcnemar_exact(a_correct, b_correct):
    """Teste exato de McNemar bicaudal sobre os casos discordantes.

    Retorna (so_a_acerta, so_b_acerta, p_valor).
    """
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))
    m = only_a + only_b
    if m == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = sum(comb(m, i) for i in range(0, k + 1)) / 2 ** m * 2
    return only_a, only_b, min(p, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Significância dos ganhos de F1 do MFE-Bot.")
    parser.add_argument("--reamostras", type=int, default=10000)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

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

    y = np.array([r[3] for r in linhas])
    pred_rf = np.array([1 if r[0] >= 0.5 else 0 for r in linhas])
    pred_gnn = np.array([1 if r[1] >= 0.5 else 0 for r in linhas])
    pred_sys = np.array([r[2] for r in linhas])

    print("=" * 72)
    print(f"Significância estatística — {len(y)} usuários, "
          f"{args.reamostras} reamostras bootstrap (semente {args.semente})")
    print("=" * 72)
    print(f"F1 pontuais: RF={f1_score(y, pred_rf):.4f}  "
          f"BotRGCN={f1_score(y, pred_gnn):.4f}  MFE-Bot={f1_score(y, pred_sys):.4f}\n")

    sys_ok = pred_sys == y
    for nome, pred, ok in [("Random Forest", pred_rf, pred_rf == y),
                           ("BotRGCN", pred_gnn, pred_gnn == y)]:
        media, lo, hi, p_le_0 = bootstrap_delta_f1(
            y, pred_sys, pred, args.reamostras, args.semente
        )
        only_sys, only_other, p_mc = mcnemar_exact(sys_ok, ok)
        sig = "SIGNIFICATIVO" if lo > 0 else "NAO significativo (IC cruza zero)"
        print(f"MFE-Bot vs {nome}:")
        print(f"  bootstrap: dF1={media:+.4f}  IC95%=[{lo:+.4f}; {hi:+.4f}]  "
              f"P(dF1<=0)={p_le_0:.4f}  -> {sig}")
        print(f"  McNemar:   so-MFE-acerta={only_sys}  so-{nome}-acerta={only_other}  "
              f"p={p_mc:.4f}\n")


if __name__ == "__main__":
    main()
