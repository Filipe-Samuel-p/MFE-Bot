"""
XAI — Seleção de Amostra Estratificada para Estudo de Caso Qualitativo.

Seleciona usuários do banco em 4 categorias investigativas (TCC §3.6.2):

    FP  Falso positivo RF    score_rf > 0.8  AND label_real = 0
    FN  Falso negativo GNN   score_gnn < 0.2 AND label_real = 1
    DR  Divergência resolvida flag_divergente AND classificacao_bot = label_real
    DF  Divergência falhada   flag_divergente AND classificacao_bot != label_real

Propósito: alimentar o XAI qualitativo (SHAP + GNNExplainer) com amostras
onde a explicação revela algo metodologicamente significativo — não gerar
explicações indiscriminadas para todo o conjunto de teste.

Uso CLI:
    python inference/xai/sample_selector.py [--limit N]

    --limit N   máximo de usuários por categoria (padrão: sem limite)

Saída: tabela formatada no stdout + dict retornado pela função `select_sample`.
"""

import argparse
import os
import sys

XAI_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(XAI_DIR)
sys.path.insert(0, INFERENCE_DIR)

from db import connect  # noqa: E402

# ---------------------------------------------------------------------------
# Critérios SQL (TCC §3.6.2, Tabela XAI).
# ---------------------------------------------------------------------------

_QUERIES = {
    "falsos_positivos_rf": {
        "label": "Falsos Positivos (RF)",
        "desc":  "RF classifica como BOT (score_rf > 0.8) mas é HUMANO",
        "sql": """
            SELECT id, user_id, screen_name,
                   score_rf, score_gnn, score_meta,
                   flag_divergente, classificacao_bot, label_real
            FROM auditoria_deteccao_bots
            WHERE score_rf > 0.8
              AND label_real = 0
            ORDER BY score_rf DESC
        """,
    },
    "falsos_negativos_gnn": {
        "label": "Falsos Negativos (GNN)",
        "desc":  "GNN classifica como HUMANO (score_gnn < 0.2) mas é BOT",
        "sql": """
            SELECT id, user_id, screen_name,
                   score_rf, score_gnn, score_meta,
                   flag_divergente, classificacao_bot, label_real
            FROM auditoria_deteccao_bots
            WHERE score_gnn < 0.2
              AND label_real = 1
            ORDER BY score_gnn ASC
        """,
    },
    "divergencias_resolvidas": {
        "label": "Divergências Resolvidas",
        "desc":  "Roteador divergiu e o meta-classificador ACERTOU",
        "sql": """
            SELECT id, user_id, screen_name,
                   score_rf, score_gnn, score_meta,
                   flag_divergente, classificacao_bot, label_real
            FROM auditoria_deteccao_bots
            WHERE flag_divergente = TRUE
              AND classificacao_bot = label_real
            ORDER BY ABS(score_meta - 0.5) ASC
        """,
    },
    "divergencias_falhadas": {
        "label": "Divergências Falhadas",
        "desc":  "Roteador divergiu e o meta-classificador ERROU",
        "sql": """
            SELECT id, user_id, screen_name,
                   score_rf, score_gnn, score_meta,
                   flag_divergente, classificacao_bot, label_real
            FROM auditoria_deteccao_bots
            WHERE flag_divergente = TRUE
              AND classificacao_bot != label_real
            ORDER BY ABS(score_meta - 0.5) ASC
        """,
    },
}

_COLS = ("id", "user_id", "screen_name",
         "score_rf", "score_gnn", "score_meta",
         "flag_divergente", "classificacao_bot", "label_real")


def select_sample(limit=None, conn=None):
    """
    Executa as 4 queries e retorna um dict:
        {categoria: [{"id": ..., "user_id": ..., ...}, ...]}

    `limit`: número máximo de linhas por categoria (None = sem limite).
    """
    close_after = conn is None
    if close_after:
        conn = connect()

    result = {}
    try:
        with conn.cursor() as cur:
            for key, meta in _QUERIES.items():
                sql = meta["sql"]
                if limit is not None:
                    sql += f" LIMIT {int(limit)}"
                cur.execute(sql)
                rows = cur.fetchall()
                result[key] = [dict(zip(_COLS, r)) for r in rows]
    finally:
        if close_after:
            conn.close()

    return result


# ---------------------------------------------------------------------------
# Formatação de tabela para o stdout.
# ---------------------------------------------------------------------------

def _fmt_score(v):
    return f"{v:.3f}" if v is not None else "  — "


def _fmt_label(v):
    if v is None:
        return " ? "
    return "BOT" if v == 1 else "HUM"


def _print_category(label, desc, rows):
    print(f"\n{'─' * 80}")
    print(f"  {label}  ({len(rows)} usuários)")
    print(f"  {desc}")
    print(f"{'─' * 80}")
    if not rows:
        print("  Nenhum usuário nesta categoria.")
        return
    header = (
        f"  {'user_id':<16} {'screen_name':<20} "
        f"{'RF':>6} {'GNN':>6} {'Meta':>6} "
        f"{'Div':>4} {'Pred':>5} {'Real':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        sn = (r["screen_name"] or "")[:20]
        print(
            f"  {r['user_id']:<16} {sn:<20} "
            f"{_fmt_score(r['score_rf']):>6} "
            f"{_fmt_score(r['score_gnn']):>6} "
            f"{_fmt_score(r['score_meta']):>6} "
            f"{'T' if r['flag_divergente'] else 'F':>4} "
            f"{_fmt_label(r['classificacao_bot']):>5} "
            f"{_fmt_label(r['label_real']):>5}"
        )


def print_sample(sample):
    print("\n" + "=" * 80)
    print("  MFE-Bot — Amostra Estratificada para XAI Qualitativo")
    print("=" * 80)
    for key, meta in _QUERIES.items():
        _print_category(meta["label"], meta["desc"], sample.get(key, []))
    print(f"\n{'=' * 80}")
    totals = {k: len(v) for k, v in sample.items()}
    print(f"  Total por categoria: {totals}")
    grand = sum(totals.values())
    print(f"  Total geral: {grand} usuários selecionados para XAI")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Seleciona amostra estratificada para XAI do MFE-Bot."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Máximo de usuários por categoria (padrão: sem limite).",
    )
    args = parser.parse_args()

    sample = select_sample(limit=args.limit)
    print_sample(sample)

    total = sum(len(v) for v in sample.values())
    if total == 0:
        print("\n[aviso] Nenhum usuário encontrado. O worker de inferência já foi executado?")


if __name__ == "__main__":
    main()
