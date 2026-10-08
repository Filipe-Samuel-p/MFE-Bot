"""
Pré-computação dos embeddings de texto (RoBERTa) para o BotRGCN.

Fiel ao BotRGCN (Feng et al., 2021a): cada nó recebe, como parte das features
iniciais, a SEMÂNTICA do seu texto, codificada por um modelo de linguagem.
Geramos, por usuário:
  - des_emb:   embedding RoBERTa da `description` do perfil
  - tweet_emb: média dos embeddings RoBERTa dos seus tweets (até ~200)

Eficiência (lições da 1ª versão, que travava em ~minutos/usuário):
  1. PADDING FIXO (`max_length`) -> todos os lotes têm o MESMO formato, então o
     MPS compila o kernel UMA vez (padding dinâmico recompilava a cada lote).
  2. BATCHING GLOBAL -> todos os tweets de todos os usuários são processados em
     lotes grandes e acumulados por usuário no próprio device (index_add_),
     sem sincronizar MPS->CPU a cada lote nem guardar 2M embeddings na RAM.

Usa MPS (GPU Metal do Apple Silicon) quando disponível. Resultado cacheado em
models/text_emb.npz e reaproveitado por graph.py (treino e worker). Roda 1×.

Uso:
    python inference/training/encode_text.py
"""

import os
import sys
import time

import ijson
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

INFERENCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, INFERENCE_DIR)
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)

from features import _clean  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_PATH = os.path.join(INFERENCE_DIR, "models", "text_emb.npz")

MODEL_NAME = "roberta-base"
EMB_DIM = 768
BATCH = 512
MAX_LEN_TWEET = 48      # cobre a maioria dos tweets; equilibra vazão/fidelidade
MAX_LEN_DES = 96
USE_FP16 = True         # autocast fp16 no device (acelera ~3.6x no MPS)
SPLITS = ("train", "dev", "test", "support")


def log(msg):
    print(msg, flush=True)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def embed_batch(texts, tok, model, device, max_len):
    """Embedding [B, 768] (float32) via mean-pooling mascarado, PADDING FIXO."""
    enc = tok(
        texts,
        padding="max_length",      # formato constante -> MPS não recompila
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    if USE_FP16 and device in ("mps", "cuda"):
        with torch.autocast(device, dtype=torch.float16):
            out = model(**enc).last_hidden_state
    else:
        out = model(**enc).last_hidden_state        # [B, T, 768]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (out.float() * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (summed / counts).float()                # float32, fica no device


def main():
    device = get_device()
    log(f"device: {device}")
    log(f"carregando {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # --- Leitura: ids, descriptions, tweets por usuário ---
    log("lendo usuários...")
    ids, des_list, user_tweets = [], [], []
    for name in SPLITS:
        path = os.path.join(DATA_DIR, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            for u in ijson.items(fh, "item"):
                uid = _clean(u.get("ID"))
                if uid is None:
                    continue
                ids.append(uid)
                prof = u.get("profile") or {}
                des_list.append(_clean(prof.get("description")) or "")
                tw = u.get("tweet") or []
                user_tweets.append([str(t) for t in tw] if tw else [])
    N = len(ids)
    log(f"usuários: {N}")

    # --- Descriptions (1 por usuário, lotes fixos) ---
    log("codificando descriptions...")
    t0 = time.time()
    des_emb = np.zeros((N, EMB_DIM), dtype=np.float32)
    for i in range(0, N, BATCH):
        emb = embed_batch(des_list[i:i + BATCH], tok, model, device, MAX_LEN_DES)
        des_emb[i:i + emb.shape[0]] = emb.cpu().numpy()
    log(f"  descriptions em {time.time()-t0:.0f}s")

    # --- Tweets: lista achatada (texto, idx_usuário) ---
    flat_texts, flat_uidx = [], []
    for idx, tweets in enumerate(user_tweets):
        for t in tweets:
            flat_texts.append(t)
            flat_uidx.append(idx)
    total = len(flat_texts)
    log(f"total de tweets a codificar: {total}")

    # Acumulação por usuário no device (sem guardar todos os embeddings).
    tweet_sum = torch.zeros(N, EMB_DIM, device=device)
    tweet_cnt = torch.zeros(N, device=device)
    t0 = time.time()
    for i in range(0, total, BATCH):
        batch = flat_texts[i:i + BATCH]
        idx = torch.tensor(flat_uidx[i:i + BATCH], device=device)
        emb = embed_batch(batch, tok, model, device, MAX_LEN_TWEET)
        tweet_sum.index_add_(0, idx, emb)
        tweet_cnt.index_add_(0, idx, torch.ones(len(batch), device=device))
        if (i // BATCH) % 200 == 0 and i > 0:
            el = time.time() - t0
            log(f"  {i}/{total} tweets | {i/el:.0f} tw/s | "
                f"{el/60:.1f} min | ETA {(total-i)/max(i/el,1)/60:.1f} min")

    tweet_emb = (tweet_sum / tweet_cnt.clamp(min=1.0).unsqueeze(1)).cpu().numpy()
    log(f"tweets em {(time.time()-t0)/60:.1f} min")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        ids=np.array(ids),
        des_emb=des_emb.astype(np.float32),
        tweet_emb=tweet_emb.astype(np.float32),
    )
    log(f"\nsalvo em: {OUT_PATH}  (des {des_emb.shape}, tweet {tweet_emb.shape})")


if __name__ == "__main__":
    main()
