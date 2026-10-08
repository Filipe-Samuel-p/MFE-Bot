"""
XAI — Explicabilidade da BotRGCN via GNNExplainer (torch_geometric.explain).

PERGUNTA QUE O GNNExplainer RESPONDE (diferente do SHAP/RF):
    "Quais conexões de rede influenciaram mais a classificação deste usuário,
     e os vizinhos relevantes são bots ou humanos?"

O SHAP (explain_rf.py) já explica FEATURES DE PERFIL.
O GNNExplainer explica ESTRUTURA DE REDE — a contribuição única da BotRGCN.

Estratégia de explicação em duas camadas:

1. ESTRUTURA DE REDE (edge_mask — principal)
   GNNExplainer aprende uma máscara escalar [0,1] por aresta sobre o grafo
   transdutivo. As arestas com maior importância são mapeadas de volta para
   user_ids, com tipo de relação (follows/followed_by) e rótulo do vizinho
   (bot/humano), revelando o padrão de homofilia que guiou a decisão.

2. FEATURES DE PERFIL (node_mask — complementar)
   O wrapper expõe apenas as 18 features de perfil ao explainer (embeddings de
   texto são fixados como buffers). A importância por feature complementa a
   visão de rede. Para texto, usa Gradient × Input separadamente.

Fallback: se `torch_geometric.explain` não estiver disponível (PyG < 2.3),
usa Gradient × Input sobre as features de perfil como substituto.

Lazy evaluation:
    1. Verifica se `explicacao_gnn` já existe no banco para o user_id.
    2. Se existir → retorna sem recalcular.
    3. Se não existir → calcula, salva com UPDATE e retorna.

Uso CLI:
    python inference/xai/explain_gnn.py <user_id> [--top-k 10]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

XAI_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(XAI_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)
sys.path.insert(0, INFERENCE_DIR)

from botrgcn import BotRGCN, normalize  # noqa: E402
from db import connect  # noqa: E402
from features import CATEGORICAL_FEATURE_NAMES, N_NUMERIC, NUMERIC_FEATURE_NAMES  # noqa: E402
from graph import REL_FOLLOWS, REL_FOLLOWED_BY, build_graph  # noqa: E402
from rgcn_flat import ExplainBotRGCN  # noqa: E402
from scoring import load_gnn  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(INFERENCE_DIR, "models")

_REL_NAME = {REL_FOLLOWS: "follows", REL_FOLLOWED_BY: "followed_by"}

try:
    from torch_geometric.explain import Explainer, GNNExplainer as _GNNExp
    _HAS_GNNEXPLAINER = True
except ImportError:
    _HAS_GNNEXPLAINER = False


# ---------------------------------------------------------------------------
# Wrapper: adapta BotRGCN à interface (x, edge_index) do GNNExplainer.
# Embeddings de texto e edge_type são buffers fixos — não são mascados.
# O explainer mascara apenas as 18 features de perfil (num + cat).
# ---------------------------------------------------------------------------

class _ProfileWrapper(torch.nn.Module):
    def __init__(self, base: BotRGCN, des: torch.Tensor, tweet: torch.Tensor,
                 edge_type: torch.Tensor, n_num: int):
        super().__init__()
        self.base = base
        self.n_num = n_num
        self.register_buffer("des", des)
        self.register_buffer("tweet", tweet)
        self.register_buffer("edge_type_buf", edge_type)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.base(
            self.des, self.tweet,
            x[:, :self.n_num], x[:, self.n_num:],
            edge_index, self.edge_type_buf,
        )


# ---------------------------------------------------------------------------
# Preparação das tensões do grafo transdutivo.
# ---------------------------------------------------------------------------

def _build_tensors(graph, bundle):
    X_num = normalize(
        graph["X_raw"][:, :N_NUMERIC],
        bundle["norm_mean"],
        bundle["norm_std"],
    )
    return {
        "des":        torch.tensor(graph["des_emb"],  dtype=torch.float32),
        "tweet":      torch.tensor(graph["tweet_emb"], dtype=torch.float32),
        "num":        torch.tensor(X_num,               dtype=torch.float32),
        "cat":        torch.tensor(graph["X_raw"][:, N_NUMERIC:], dtype=torch.float32),
        "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
        "edge_type":  torch.tensor(graph["edge_type"],  dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 1. Explicação de REDE via edge_mask do GNNExplainer (principal).
# ---------------------------------------------------------------------------

def _extract_top_neighbors(edge_mask, edge_index, edge_type, node_idx,
                            graph, top_k):
    """
    Mapeia as arestas mais importantes para user_ids e rótulos de vizinhos.

    Considera apenas arestas onde o nó alvo é source OU target (vizinhança
    direta de 1 hop). Isso dá a resposta mais interpretável:
    "estas são as conexões diretas que mais pesaram na decisão".
    """
    ei = edge_index.cpu().numpy()   # (2, E)
    et = edge_type.cpu().numpy()    # (E,)
    em = edge_mask.cpu().numpy()    # (E,)

    # Filtra arestas que envolvem diretamente o nó alvo.
    mask_direct = (ei[0] == node_idx) | (ei[1] == node_idx)
    indices_direct = np.where(mask_direct)[0]

    if len(indices_direct) == 0:
        return [], {}

    # Ordena por importância decrescente e pega top_k.
    order = indices_direct[np.argsort(em[indices_direct])[::-1]][:top_k]

    ids = graph["ids"]
    y = graph["y"]

    top_neighbors = []
    bot_count = 0
    human_count = 0

    for idx in order:
        src, dst = int(ei[0, idx]), int(ei[1, idx])
        rel = _REL_NAME.get(int(et[idx]), "unknown")
        importance = round(float(em[idx]), 4)

        neighbor_idx = dst if src == node_idx else src
        neighbor_uid = ids[neighbor_idx]
        raw_label = int(y[neighbor_idx])
        neighbor_label = raw_label if raw_label >= 0 else None

        if neighbor_label == 1:
            bot_count += 1
        elif neighbor_label == 0:
            human_count += 1

        top_neighbors.append({
            "neighbor_user_id": neighbor_uid,
            "relation": rel,
            "edge_importance": importance,
            "neighbor_label": neighbor_label,
        })

    total_labeled = bot_count + human_count
    edge_summary = {
        "direct_edges_total": int(len(indices_direct)),
        "num_important_direct_edges": int(len(order)),
        "bot_neighbors_ratio": round(bot_count / total_labeled, 4) if total_labeled > 0 else None,
        "bot_neighbors_count": bot_count,
        "human_neighbors_count": human_count,
    }
    return top_neighbors, edge_summary


def _run_gnnexplainer(wrapper, x_profile, edge_index, edge_type_t, node_idx,
                      graph, top_k, epochs=200):
    """
    Executa GNNExplainer e retorna a explicação de rede + perfil.

    Retorna:
        top_neighbors   list — vizinhos mais importantes com metadados
        edge_summary    dict — estatísticas de rede
        profile_imp     np.ndarray (18,) — importância das features de perfil
    """
    exp = Explainer(
        model=wrapper,
        algorithm=_GNNExp(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    explanation = exp(x_profile, edge_index, index=int(node_idx))

    # --- Explicação de rede (edge_mask) ---
    em = explanation.edge_mask.detach()
    top_neighbors, edge_summary = _extract_top_neighbors(
        em, edge_index, edge_type_t, node_idx, graph, top_k
    )

    # --- Importância de perfil (node_mask) ---
    nm = explanation.node_mask.detach()
    if nm.dim() == 2 and nm.size(0) == x_profile.size(0):
        profile_imp = nm[node_idx].cpu().numpy()
    else:
        profile_imp = nm.squeeze(0).cpu().numpy()

    return top_neighbors, edge_summary, profile_imp


# ---------------------------------------------------------------------------
# 2. Gradient × Input — fallback e importância do texto.
# ---------------------------------------------------------------------------

def _gradient_attribution(gnn_model, tensors, node_idx):
    """
    Gradient × Input: |grad * value| para cada dimensão de cada modalidade.
    Retorna (num_attr, cat_attr, des_scalar, tweet_scalar).
    """
    des   = tensors["des"].clone().detach().requires_grad_(True)
    tweet = tensors["tweet"].clone().detach().requires_grad_(True)
    num   = tensors["num"].clone().detach().requires_grad_(True)
    cat   = tensors["cat"].clone().detach().requires_grad_(True)

    logits = gnn_model(des, tweet, num, cat,
                       tensors["edge_index"], tensors["edge_type"])
    torch.softmax(logits, dim=1)[node_idx, 1].backward()

    return (
        (num.grad[node_idx]   * num[node_idx]).abs().detach().numpy(),
        (cat.grad[node_idx]   * cat[node_idx]).abs().detach().numpy(),
        float((des.grad[node_idx]   * des[node_idx]).abs().sum().item()),
        float((tweet.grad[node_idx] * tweet[node_idx]).abs().sum().item()),
    )


def _norm01(arr):
    m = arr.max()
    return arr / m if m > 0 else arr


# ---------------------------------------------------------------------------
# Cache em memória (nível de processo): construir o grafo transdutivo exige
# ler e parsear train/dev/test.json (~12 mil usuários) e realinhar os
# embeddings de texto (~61 MB) a cada chamada. Isso é caro (dezenas de
# segundos) e desnecessário quando várias explicações são pedidas na mesma
# sessão (ex.: notebook, laço sobre uma amostra estratificada). O cache vive
# apenas enquanto o processo Python estiver de pé — cada novo processo (nova
# invocação de CLI) reconstrói uma vez, o que é aceitável por ser um custo
# único por execução.
# ---------------------------------------------------------------------------

_resource_cache = {"gnn_model": None, "bundle": None, "graph": None}


def _get_cached_resources():
    if _resource_cache["graph"] is None:
        _resource_cache["gnn_model"], _resource_cache["bundle"] = load_gnn(
            os.path.join(MODELS_DIR, "modelo_gnn.pt")
        )
        _resource_cache["graph"] = build_graph(DATA_DIR)
    return (
        _resource_cache["gnn_model"],
        _resource_cache["bundle"],
        _resource_cache["graph"],
    )


# ---------------------------------------------------------------------------
# Função pública principal.
# ---------------------------------------------------------------------------

def compute_gnn_explanation(user_id, gnn_model=None, bundle=None, graph=None,
                             top_k=10, epochs=200):
    """
    Calcula a explicação GNN para um usuário e devolve um dict JSON-serializável.

    Campos principais do dict (quando GNNExplainer disponível):
        method                    "GNNExplainer" ou "GradientAttribution"
        prediction                float — probabilidade bot (softmax do GNN)
        network_explanation       dict — foco principal: ESTRUTURA DE REDE
            top_neighbors         list[dict] — vizinhos diretos mais importantes
                .neighbor_user_id str
                .relation         "follows" | "followed_by"
                .edge_importance  float [0,1]
                .neighbor_label   0 (humano) | 1 (bot) | null (sem label)
            edge_summary          dict — estatísticas de rede
                .direct_edges_total
                .bot_neighbors_ratio  — indicador de homofilia
                .bot_neighbors_count
                .human_neighbors_count
        profile_feature_importance  list[dict] — complementar (SHAP cobre melhor)
            .feature, .importance, .raw_value
        modality_summary          dict — fração de importância por modalidade
        computed_at               ISO 8601 UTC
    """
    if gnn_model is None or bundle is None or graph is None:
        cached_model, cached_bundle, cached_graph = _get_cached_resources()
        gnn_model = gnn_model if gnn_model is not None else cached_model
        bundle = bundle if bundle is not None else cached_bundle
        graph = graph if graph is not None else cached_graph

    uid = str(user_id).strip()
    node_idx = graph["id_to_idx"].get(uid)
    if node_idx is None:
        raise LookupError(f"Usuário {user_id!r} não encontrado no grafo transdutivo.")

    tensors = _build_tensors(graph, bundle)

    # Predição bruta (p_gnn).
    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(
            tensors["des"], tensors["tweet"],
            tensors["num"], tensors["cat"],
            tensors["edge_index"], tensors["edge_type"],
        )
        p_bot = float(torch.softmax(logits, dim=1)[node_idx, 1].item())

    # --- Importância de texto via Gradient × Input (sempre calculada) ---
    num_grad, cat_grad, des_scalar, tweet_scalar = _gradient_attribution(
        gnn_model, tensors, node_idx
    )

    # --- Explicação principal: REDE (GNNExplainer) ---
    top_neighbors = []
    edge_summary = {}
    method = "GradientAttribution"
    profile_imp_raw = None

    if _HAS_GNNEXPLAINER:
        try:
            # A RGCNConv propaga uma vez por relação, o que é incompatível com o
            # mascaramento de arestas do GNNExplainer. ExplainBotRGCN reproduz o
            # mesmo forward (mesmos pesos) numa única propagação, tornando a rede
            # explicável. Ver inference/xai/rgcn_flat.py.
            explain_model = ExplainBotRGCN(gnn_model).eval()
            wrapper = _ProfileWrapper(
                explain_model,
                tensors["des"], tensors["tweet"], tensors["edge_type"],
                N_NUMERIC,
            )
            x_profile = torch.cat([tensors["num"], tensors["cat"]], dim=1)
            top_neighbors, edge_summary, profile_imp_raw = _run_gnnexplainer(
                wrapper, x_profile, tensors["edge_index"], tensors["edge_type"],
                node_idx, graph, top_k, epochs,
            )
            method = "GNNExplainer"
        except Exception as exc:
            print(f"[warn] GNNExplainer falhou ({type(exc).__name__}: {exc}); "
                  f"usando GradientAttribution.", file=sys.stderr)

    # --- Importância de perfil (node_mask do GNNExplainer ou Gradient × Input) ---
    if profile_imp_raw is not None:
        num_imp = _norm01(profile_imp_raw[:N_NUMERIC])
        cat_imp = _norm01(profile_imp_raw[N_NUMERIC:])
    else:
        num_imp = _norm01(num_grad)
        cat_imp = _norm01(cat_grad)

    raw_num = graph["X_raw"][node_idx, :N_NUMERIC]
    raw_cat = graph["X_raw"][node_idx, N_NUMERIC:]

    profile_feature_importance = []
    for i, name in enumerate(NUMERIC_FEATURE_NAMES):
        profile_feature_importance.append({
            "feature":    name,
            "importance": round(float(num_imp[i]), 4),
            "raw_value":  round(float(raw_num[i]), 4),
        })
    for i, name in enumerate(CATEGORICAL_FEATURE_NAMES):
        profile_feature_importance.append({
            "feature":    name,
            "importance": round(float(cat_imp[i]), 4),
            "raw_value":  round(float(raw_cat[i]), 4),
        })
    profile_feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    # Resumo de importância por modalidade (Gradient × Input, comparável entre métodos).
    sum_num  = float(np.sum(num_grad))
    sum_cat  = float(np.sum(cat_grad))
    total    = sum_num + sum_cat + des_scalar + tweet_scalar
    if total > 0:
        modality_summary = {
            "description_text":    round(des_scalar  / total, 4),
            "tweets_text":         round(tweet_scalar / total, 4),
            "numeric_profile":     round(sum_num      / total, 4),
            "categorical_profile": round(sum_cat      / total, 4),
        }
    else:
        modality_summary = {k: 0.0 for k in
                            ("description_text", "tweets_text",
                             "numeric_profile", "categorical_profile")}

    result = {
        "method":     method,
        "prediction": round(p_bot, 6),
        "network_explanation": {
            "top_neighbors": top_neighbors,
            "edge_summary":  edge_summary,
        },
        "profile_feature_importance": profile_feature_importance,
        "modality_summary": modality_summary,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


# ---------------------------------------------------------------------------
# Persistência.
# ---------------------------------------------------------------------------

def save_to_db(user_id, explanation, conn=None):
    """Grava explicacao_gnn no banco via UPDATE (idempotente)."""
    close_after = conn is None
    if close_after:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auditoria_deteccao_bots SET explicacao_gnn = %s WHERE user_id = %s",
                (json.dumps(explanation, ensure_ascii=False), str(user_id)),
            )
        conn.commit()
    finally:
        if close_after:
            conn.close()


def explain_lazy(user_id, conn=None, gnn_model=None, bundle=None, graph=None,
                 top_k=10):
    """
    Lazy evaluation: calcula e persiste apenas se ainda não existe no banco.
    Retorna o dict de explicação (seja do cache ou recém-calculado).
    """
    close_after = conn is None
    if close_after:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT explicacao_gnn FROM auditoria_deteccao_bots WHERE user_id = %s",
                (str(user_id),),
            )
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"user_id={user_id!r} não encontrado no banco.")
        if row[0] is not None:
            return row[0]
        exp = compute_gnn_explanation(user_id, gnn_model=gnn_model,
                                      bundle=bundle, graph=graph, top_k=top_k)
        save_to_db(user_id, exp, conn=conn)
        return exp
    finally:
        if close_after:
            conn.close()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calcula explicação GNN (estrutura de rede) para um usuário."
    )
    parser.add_argument("user_id", help="user_id do TwiBot-20")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Número de vizinhos mais importantes a reportar (padrão: 10).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Épocas de otimização do GNNExplainer (padrão: 200).")
    args = parser.parse_args()

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT explicacao_gnn FROM auditoria_deteccao_bots WHERE user_id = %s",
                (args.user_id,),
            )
            row = cur.fetchone()

        if row is None:
            sys.exit(f"user_id={args.user_id!r} não encontrado na tabela auditoria_deteccao_bots.")

        if row[0] is not None:
            print(f"[cache] Explicação GNN já existe para user_id={args.user_id!r}:")
            print(json.dumps(row[0], indent=2, ensure_ascii=False))
            return

        print(f"Construindo grafo e executando GNNExplainer para user_id={args.user_id!r}...")
        exp = compute_gnn_explanation(args.user_id, top_k=args.top_k, epochs=args.epochs)
        save_to_db(args.user_id, exp, conn=conn)
        print(f"Método: {exp['method']}")
        print("Salvo em explicacao_gnn:")
        print(json.dumps(exp, indent=2, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
