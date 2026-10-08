"""
Fase 2 — Worker de inferência (orquestrador do pipeline online).

Processa o `test.json` e popula o PostgreSQL (fonte única da verdade):
1. Carrega os 3 modelos congelados (RF, BotRGCN, LR) via scoring.py.
2. Constrói o grafo transdutivo (graph.py, streaming ijson).
3. Pontua TODOS os nós: RF (por usuário) + BotRGCN (passada única no grafo).
4. Para cada usuário do test.json, aplica o Roteamento por Cruzamento de
   Fronteira (router.py) e faz INSERT com todos os campos + label_real.

Nota transdutiva: o GNN não pontua por usuário em streaming — roda uma vez sobre
o grafo inteiro e o worker lê o p_gnn de cada nó de test.

Uso (worker LOCAL contra o Postgres do docker):
    python inference/worker/run_inference.py
"""

import os
import sys

import ijson
from psycopg2.extras import execute_values

WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(WORKER_DIR)
sys.path.insert(0, INFERENCE_DIR)
sys.path.insert(0, WORKER_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)

from db import connect  # noqa: E402
from features import _clean  # noqa: E402
from graph import build_graph  # noqa: E402
from router import route  # noqa: E402
from scoring import gnn_scores, load_gnn, load_rf, rf_scores  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(INFERENCE_DIR, "models")
TEST_PATH = os.path.join(DATA_DIR, "test.json")
SPLIT_TEST = 2
BATCH = 500

INSERT_SQL = """
INSERT INTO auditoria_deteccao_bots
  (user_id, screen_name, score_rf, score_gnn, score_meta,
   flag_divergente, rf_only, veredito_final, classificacao_bot, label_real)
VALUES %s
"""


def load_screen_names(path):
    """Mapa ID -> screen_name (o grafo não guarda screen_name, só features)."""
    names = {}
    with open(path, "rb") as fh:
        for u in ijson.items(fh, "item"):
            uid = _clean(u.get("ID"))
            if uid is None:
                continue
            prof = u.get("profile") or {}
            names[uid] = _clean(prof.get("screen_name"))
    return names


def main():
    print("=" * 60)
    print("Worker de inferência — MFE-Bot (Fase 2)")
    print("=" * 60)

    print("\nCarregando modelos e construindo grafo...")
    rf = load_rf(os.path.join(MODELS_DIR, "modelo_rf.pkl"))
    gnn, bundle = load_gnn(os.path.join(MODELS_DIR, "modelo_gnn.pt"))
    import joblib
    lr = joblib.load(os.path.join(MODELS_DIR, "modelo_lr.pkl"))

    g = build_graph(DATA_DIR)
    print("Pontuando (RF por usuário + BotRGCN no grafo inteiro)...")
    p_rf_all = rf_scores(rf, g["X_raw"])
    p_gnn_all = gnn_scores(gnn, bundle, g)
    screen = load_screen_names(TEST_PATH)

    # Monta as linhas dos usuários de test.
    rows = []
    n_conv = n_div = n_rfonly = 0
    for i in range(len(g["ids"])):
        if g["split"][i] != SPLIT_TEST:
            continue
        uid = g["ids"][i]
        p_rf = float(p_rf_all[i])
        p_gnn = float(p_gnn_all[i])
        has_nb = bool(g["has_neighbor"][i])
        r = route(p_rf, p_gnn, lr, has_neighbor=has_nb)
        label = int(g["y"][i]) if g["y"][i] >= 0 else None

        if r.rf_only:
            n_rfonly += 1
        elif r.flag_divergente:
            n_div += 1
        else:
            n_conv += 1

        rows.append((
            uid, screen.get(uid), p_rf, p_gnn, r.score_meta,
            r.flag_divergente, r.rf_only, r.veredito_final, r.classificacao_bot, label,
        ))

    print(f"\nUsuários de test: {len(rows)}")
    print(f"  convergentes: {n_conv} | divergentes: {n_div} | RF-only (neighbor:null): {n_rfonly}")

    # Persiste (idempotente: TRUNCATE + INSERT em lote).
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE auditoria_deteccao_bots RESTART IDENTITY;")
            for i in range(0, len(rows), BATCH):
                execute_values(cur, INSERT_SQL, rows[i:i + BATCH])
        conn.commit()
        print(f"\nGravadas {len(rows)} linhas em auditoria_deteccao_bots.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
