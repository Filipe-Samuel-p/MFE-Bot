"""
Construção do grafo heterogêneo do TwiBot-20 para a BotRGCN.

Módulo COMPARTILHADO entre o treino (training/train_gnn.py) e o worker de
inferência (worker/run_inference.py). O grafo é TRANSDUTIVO: um único grafo
com todos os usuários rotulados (train+dev+test), conforme o desenho
semi-supervisionado do TwiBot-20 (FAQ oficial: "If one user follows another,
there is an edge between them"). O modelo treina apenas com a máscara de
treino; os labels de dev/test nunca entram na loss.

Particularidades tratadas:
- A chave do nó é o `ID` de topo (limpo), não `profile.id` (tem espaço).
- Arestas só existem entre usuários PRESENTES no dataset; vizinhos externos
  (support set, ausente aqui) são ignorados.
- `neighbor` pode ser None e as listas `following`/`follower` podem ser vazias.
- Se existir `data/support.json`, seus usuários entram como nós de suporte
  (sem label) para densificar o grafo — opcional, funciona sem ele.

Saída: um dicionário com arrays NumPy/listas puras (sem dependência de torch),
para que o treino e o worker montem o objeto PyG e apliquem a normalização.
"""

import os

import ijson
import numpy as np

from features import FEATURE_NAMES, extract_features, parse_label

# Tipos de aresta (relações da RGCN), DIRECIONADAS — preserva a direção das
# conexões (TCC §3.3.1), sem simetrizar. Cada relação tem matriz de peso
# própria na RGCN; nós sem arestas de entrada mantêm o sinal via root weight.
REL_FOLLOWS = 0       # u -> f : u segue f          (following[u])
REL_FOLLOWED_BY = 1   # u -> f : u é seguido por f  (follower[u])
NUM_RELATIONS = 2

TEXT_EMB_FILE = "text_emb.npz"  # cache gerado por training/encode_text.py

# Splits supervisionados + suporte opcional.
SUPERVISED_SPLITS = ("train", "dev", "test")


def _clean_id(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def build_graph(data_dir, include_support=True):
    """
    Lê os arquivos do TwiBot-20 e devolve a estrutura do grafo transdutivo.

    Retorna um dict com:
      - ids:        lista de IDs (str) na ordem dos índices dos nós
      - id_to_idx:  mapa ID -> índice
      - X_raw:      np.float64 [N, F]  features de perfil (não normalizadas)
      - y:          np.int64   [N]     label (0/1) ou -1 se desconhecido (suporte)
      - split:      np.int8    [N]     0=train 1=dev 2=test 3=support
      - edge_index: np.int64   [2, E]
      - edge_type:  np.int64   [E]     REL_FOLLOWS / REL_FOLLOWED_BY
      - has_neighbor: np.bool_ [N]     True se o usuário tinha neighbor != null
      - feature_names: lista de nomes das colunas
    """
    files = []
    for i, name in enumerate(SUPERVISED_SPLITS):
        files.append((os.path.join(data_dir, f"{name}.json"), i))
    support_path = os.path.join(data_dir, "support.json")
    if include_support and os.path.exists(support_path):
        files.append((support_path, 3))

    ids = []
    id_to_idx = {}
    X_raw = []
    y = []
    split = []
    has_neighbor = []
    # Guardamos as listas de vizinhos para uma 2ª passada (precisamos do
    # conjunto completo de nós antes de resolver arestas internas).
    raw_neighbors = []  # (uid, following[], follower[])

    for path, split_id in files:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            for user in ijson.items(fh, "item"):
                uid = _clean_id(user.get("ID"))
                if uid is None or uid in id_to_idx:
                    continue
                idx = len(ids)
                id_to_idx[uid] = idx
                ids.append(uid)
                X_raw.append(extract_features(user))
                label = parse_label(user)
                y.append(label if label is not None else -1)
                split.append(split_id)

                nb = user.get("neighbor")
                if isinstance(nb, dict):
                    has_neighbor.append(True)
                    following = [_clean_id(x) for x in (nb.get("following") or [])]
                    follower = [_clean_id(x) for x in (nb.get("follower") or [])]
                    raw_neighbors.append((uid, following, follower))
                else:
                    has_neighbor.append(False)

    # 2ª passada: arestas DIRECIONADAS, apenas entre nós existentes.
    src, dst, etype = [], [], []

    def add_edge(a, b, rel):
        src.append(a); dst.append(b); etype.append(rel)

    for uid, following, follower in raw_neighbors:
        u = id_to_idx[uid]
        for f in following:
            j = id_to_idx.get(f)
            if j is not None:
                add_edge(u, j, REL_FOLLOWS)
        for f in follower:
            j = id_to_idx.get(f)
            if j is not None:
                add_edge(u, j, REL_FOLLOWED_BY)

    edge_index = (
        np.asarray([src, dst], dtype=np.int64)
        if src
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_type = np.asarray(etype, dtype=np.int64)

    # Embeddings de texto (RoBERTa) alinhados à ordem dos nós, se já cacheados.
    des_emb, tweet_emb = _load_text_embeddings(ids)

    return {
        "ids": ids,
        "id_to_idx": id_to_idx,
        "X_raw": np.asarray(X_raw, dtype=np.float64),
        "des_emb": des_emb,
        "tweet_emb": tweet_emb,
        "y": np.asarray(y, dtype=np.int64),
        "split": np.asarray(split, dtype=np.int8),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "has_neighbor": np.asarray(has_neighbor, dtype=np.bool_),
        "feature_names": list(FEATURE_NAMES),
    }


def _load_text_embeddings(ids):
    """Carrega models/text_emb.npz e reordena os embeddings para a ordem `ids`.

    Devolve (des_emb, tweet_emb) como np.float32 [N, 768]. Se o cache não
    existir, devolve None em ambos (o treino do GNN exigirá rodar encode_text).
    """
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    path = os.path.join(models_dir, TEXT_EMB_FILE)
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True)
    emb_ids = [str(x) for x in data["ids"]]
    pos = {uid: i for i, uid in enumerate(emb_ids)}
    dim = data["des_emb"].shape[1]
    des = np.zeros((len(ids), dim), dtype=np.float32)
    tweet = np.zeros((len(ids), dim), dtype=np.float32)
    for i, uid in enumerate(ids):
        j = pos.get(uid)
        if j is not None:
            des[i] = data["des_emb"][j]
            tweet[i] = data["tweet_emb"][j]
    return des, tweet
